import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin


class PromptCodebookTuner(PreTrainedModel):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        num_prompts,
        prompt_len,
        forbidden_token_ids=None,
        dissim_coef=0.1,
        special_token_coef=0.1,
        role="system",
    ):
        super().__init__(config=model.config)

        # for compatibility with trainer
        self.supports_gradient_checkpointing = model.supports_gradient_checkpointing
        self._supports_sdpa = model._supports_sdpa
        self._supports_flash_attn_2 = model._supports_flash_attn_2
        self._supports_flex_attn = model._supports_flex_attn
        self._is_hf_initialized = True
        self._is_stateful = model._is_stateful
        self.gradient_checkpointing = False

        self.model = model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.num_prompts = num_prompts
        self.dissim_coef = dissim_coef
        self.special_token_coef = special_token_coef

        # Проверяем наличие chat template
        if not self.tokenizer.chat_template:
            raise ValueError("Tokenizer must have a chat_template defined.")

        # Получаем префикс и суффикс для указанной роли
        prefix_tokens, suffix_tokens = self._get_template_parts(role)
        self.register_buffer("prefix_tokens", prefix_tokens)
        self.register_buffer("suffix_tokens", suffix_tokens)

        # Замораживаем основную модель
        for param in model.parameters():
            param.requires_grad = False

        # Настройка запрещенных токенов
        self.forbidden_token_ids = forbidden_token_ids or []
        self.forbidden_token_ids = list(set(self.forbidden_token_ids))

        # Получаем эмбеддинги модели
        embedding_layer = model.get_input_embeddings()
        self.emb_dim = embedding_layer.embedding_dim

        # Инициализация кодбука ближе к реальным эмбеддингам
        init_emb = embedding_layer.weight.mean(dim=0)
        self.codebook = nn.Parameter(
            torch.randn(num_prompts, prompt_len, self.emb_dim).to(dtype=init_emb.dtype, device=init_emb.device) * 0.1 + init_emb
        )

        # Регистрируем эмбеддинги словаря
        self.register_buffer("vocab_embeddings", embedding_layer.weight.detach())

    def _get_template_parts(self, role):
        """Получаем префикс и суффикс токены для указанной роли"""
        # Генерируем тестовые токены с уникальным контентом
        test_content = "ABCDEFGH"
        messages = [{"role": role, "content": test_content}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, return_tensors="pt"
            )[0].tolist()
        except Exception as e:
            raise ValueError(
                f"Error applying chat template for role '{role}': {str(e)}"
            )

        # Токенизируем тестовый контент отдельно
        content_tokens = self.tokenizer.encode(test_content, add_special_tokens=False)

        # Ищем позицию контента в отформатированных токенах
        start_idx = None
        for i in range(len(formatted) - len(content_tokens) + 1):
            if formatted[i : i + len(content_tokens)] == content_tokens:
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("Could not determine template parts")

        prefix = formatted[:start_idx]
        suffix = formatted[start_idx + len(content_tokens) :]

        return torch.tensor(prefix, dtype=torch.long), torch.tensor(
            suffix, dtype=torch.long
        )

    def _quantize_codebook(self):
        """Квантование с straight-through estimator"""
        flat_codebook = self.codebook.view(-1, self.emb_dim)

        # Рассчитываем расстояния до всех токенов
        distances = torch.cdist(flat_codebook, self.vocab_embeddings)
        probs = F.softmax(-distances, dim=-1)  # Для мягкого штрафа

        # Находим ближайшие токены
        nearest_indices = distances.argmin(dim=-1)
        quantized_embeddings = self.model.get_input_embeddings()(nearest_indices)

        # Straight-through estimator
        quantized_embeddings = (
            self.codebook
            + (quantized_embeddings.view_as(self.codebook) - self.codebook).detach()
        )

        return (
            quantized_embeddings,
            nearest_indices.view(self.num_prompts, self.prompt_len),
            probs,
        )

    def _calculate_aux_losses(self, probs):
        """Дополнительные функции потерь"""
        # Потеря на разнообразие промптов
        mean_prompts = self.codebook.mean(dim=1)
        sim_matrix = F.cosine_similarity(
            mean_prompts.unsqueeze(1), mean_prompts.unsqueeze(0), dim=-1
        )
        dissim_loss = (sim_matrix.triu(diagonal=1)).mean()

        # Штраф за использование запрещенных токенов
        if len(self.forbidden_token_ids) > 0:
            forbidden_probs = probs[:, self.forbidden_token_ids].sum(dim=-1)
            special_penalty = forbidden_probs.mean()
        else:
            special_penalty = 0

        return dissim_loss, special_penalty

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)

        prompt_embeddings, prompt_indices, probs = self._quantize_codebook()
        dissim_loss, special_penalty = self._calculate_aux_losses(probs)

        prefix_emb = self.model.get_input_embeddings()(
            self.prefix_tokens.to(prompt_embeddings.device)
        )
        suffix_emb = self.model.get_input_embeddings()(
            self.suffix_tokens.to(prompt_embeddings.device)
        )

        full_embs = []
        for i in range(self.num_prompts):
            full_embs.append(torch.cat([prefix_emb, prompt_embeddings[i], suffix_emb]))
        full_prompt_embeddings = torch.stack(full_embs)

        prefix_len = self.prefix_tokens.size(0)
        suffix_len = self.suffix_tokens.size(0)

        all_logits, all_losses = [], []
        for i in range(self.num_prompts):
            current_prompt_emb = full_prompt_embeddings[i]
            prompt_emb_batch = current_prompt_emb.unsqueeze(0).expand(
                batch_size, -1, -1
            )

            input_emb = self.model.get_input_embeddings()(input_ids)

            combined_emb = torch.cat([prompt_emb_batch, input_emb], dim=1)

            # Создаем маску с учетом pad_token_id в кодбуке
            codebook_tokens = prompt_indices[i].to(
                input_ids.device
            )  # Токены кодбука для текущего промпта
            codebook_mask = (
                codebook_tokens != self.tokenizer.pad_token_id
            ).long()  # 1 где не pad, 0 где pad

            # Собираем полную маску: префикс + кодбук + суффикс
            current_prompt_mask = torch.cat(
                [
                    torch.ones(prefix_len, dtype=torch.long, device=input_ids.device),
                    codebook_mask,
                    torch.ones(suffix_len, dtype=torch.long, device=input_ids.device),
                ]
            )

            prompt_mask_batch = current_prompt_mask.unsqueeze(0).expand(batch_size, -1)

            if attention_mask is not None:
                combined_mask = torch.cat([prompt_mask_batch, attention_mask], dim=1)
            else:
                combined_mask = prompt_mask_batch

            outputs = self.model(
                inputs_embeds=combined_emb.contiguous(),
                attention_mask=combined_mask.contiguous(),
                **kwargs,
            )

            all_logits.append(outputs.logits)
            if outputs.loss is not None:
                all_losses.append(outputs.loss)

        logits = (
            torch.cat(all_logits, dim=0)
            .reshape(self.num_prompts, *all_logits[0].shape)
            .contiguous()
        )
        losses = torch.stack(all_losses).contiguous() if all_losses else None

        aux_loss = (
            self.dissim_coef * dissim_loss + self.special_token_coef * special_penalty
        )

        return {"logits": logits, "losses": losses, "aux_loss": aux_loss}

    def get_codebook_tokens(self, return_strings=True):
        """
        Возвращает промпты с чат-темплейтом в виде токенов
        """
        with torch.no_grad():
            _, prompt_indices, _ = self._quantize_codebook()

            full_prompts = []
            for prompt in prompt_indices:
                # Добавляем префикс и суффикс
                full_tokens = (
                    torch.cat(
                        [
                            self.prefix_tokens.to(prompt.device),
                            prompt,
                            self.suffix_tokens.to(prompt.device),
                        ]
                    )
                    .cpu()
                    .numpy()
                )

                if return_strings:
                    full_prompts.append(
                        self.tokenizer.decode(full_tokens, skip_special_tokens=False)
                    )
                else:
                    full_prompts.append(
                        self.tokenizer.convert_ids_to_tokens(full_tokens)
                    )

            return {
                "prompts": full_prompts,
                "tokens": prompt_indices.cpu().numpy().tolist(),
            }
