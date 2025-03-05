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
        init_prompt=None,
        fused_forward=True,
        gumbel_temp=0.05,
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
        self.init_prompt = init_prompt
        self.fused_forward = fused_forward
        self.gumbel_temp = gumbel_temp

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

        init_noise_coef = 0.01

        # Инициализация кодбука с учетом текстового промпта
        if init_prompt is not None:
            # Токенизация с учетом ограничений длины
            encoding = tokenizer(
                init_prompt,
                # add_special_tokens=False,
                max_length=prompt_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0).to(model.device)

            # Получение эмбеддингов токенов
            with torch.no_grad():
                base_embeddings = embedding_layer(input_ids)

            # Повторение и добавление шума
            codebook_init = (
                base_embeddings.clone().unsqueeze(0).repeat(num_prompts, 1, 1)
            )
            codebook_init += (
                torch.randn_like(codebook_init).to(
                    dtype=codebook_init.dtype, device=codebook_init.device
                )
                * init_noise_coef
            )
            self.codebook = nn.Parameter(codebook_init)
        else:
            # Стандартная инициализация
            init_emb = embedding_layer.weight.mean(dim=0)
            self.codebook = nn.Parameter(
                torch.randn(num_prompts, prompt_len, self.emb_dim).to(
                    dtype=init_emb.dtype, device=init_emb.device
                )
                * init_noise_coef
                + init_emb
            )

        self.noise_scale = nn.Parameter(
            torch.tensor(0.05).to(
                dtype=embedding_layer.weight.dtype, device=embedding_layer.weight.device
            )
        )  # Initial scale factor

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

    def _quantize_codebook(self, training=True):
        """Quantization with corrected Gumbel-Softmax and straight-through estimator."""
        flat_codebook = self.codebook.view(-1, self.emb_dim)

        # Isolate vocabulary embeddings with clone
        safe_vocab_emb = self.vocab_embeddings.clone().detach()

        # Compute logits as dot products (no normalization)
        codebook_norm = F.normalize(flat_codebook, p=2, dim=-1)
        vocab_norm = F.normalize(safe_vocab_emb, p=2, dim=-1)

        # Scaled cosine similarity
        logits = torch.matmul(
            flat_codebook, safe_vocab_emb.T
        )  # [num_prompts * seq_len, vocab_size]

        # # Mask forbidden tokens
        # if self.forbidden_token_ids:
        #     logits[:, self.forbidden_token_ids] = -torch.finfo(logits.dtype).max

        # Gumbel-Softmax with adaptive temperature
        if training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
            scaled_noise = gumbel_noise * self.noise_scale
            noisy_logits = (logits + scaled_noise) / self.gumbel_temp
            probs = F.softmax(noisy_logits, dim=-1)
        else:
            probs = F.one_hot(logits.argmax(dim=-1), logits.size(-1)).to(
                dtype=self.vocab_embeddings.dtype
            )

        # Compute soft and hard quantized embeddings
        soft_quant = torch.matmul(probs, safe_vocab_emb)
        hard_indices = probs.argmax(dim=-1)
        hard_quant = safe_vocab_emb[hard_indices]

        # Straight-Through Estimator
        if training:
            quantized_embeddings = hard_quant + (soft_quant - soft_quant.detach())
        else:
            quantized_embeddings = hard_quant

        # Reshape to original dimensions
        quantized_embeddings = quantized_embeddings.view(
            self.num_prompts, self.prompt_len, -1
        )
        hard_indices = hard_indices.view(self.num_prompts, self.prompt_len)

        return quantized_embeddings, hard_indices, probs

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

        # Квантование кодбука
        prompt_embeddings, prompt_indices, probs = self._quantize_codebook(
            training=self.training
        )
        dissim_loss, special_penalty = self._calculate_aux_losses(probs)

        # Получаем префикс и суффикс
        prefix_emb = self.model.get_input_embeddings()(
            self.prefix_tokens.to(prompt_embeddings.device)
        )
        suffix_emb = self.model.get_input_embeddings()(
            self.suffix_tokens.to(prompt_embeddings.device)
        )
        prefix_len, suffix_len = prefix_emb.size(0), suffix_emb.size(0)

        # Собираем полные промпты для всех промптов
        full_prompt_embeddings = torch.cat(
            [
                prefix_emb.unsqueeze(0).expand(self.num_prompts, -1, -1),
                prompt_embeddings,
                suffix_emb.unsqueeze(0).expand(self.num_prompts, -1, -1),
            ],
            dim=1,
        )  # [num_prompts, total_prompt_len, emb_dim]

        if self.fused_forward:
            # Подготавливаем маски для кодбука
            codebook_mask = (prompt_indices != self.tokenizer.pad_token_id).long()
            prompt_mask = torch.cat(
                [
                    torch.ones(self.num_prompts, prefix_len, device=self.device),
                    codebook_mask,
                    torch.ones(self.num_prompts, suffix_len, device=self.device),
                ],
                dim=1,
            )  # [num_prompts, total_prompt_len]

            # Векторизованное расширение для батча
            total_prompt_len = full_prompt_embeddings.size(1)
            input_emb = self.model.get_input_embeddings()(input_ids)
            seq_len = input_emb.size(1)

            # Собираем все эмбеддинги
            combined_emb = torch.cat(
                [
                    full_prompt_embeddings.unsqueeze(1)
                    .expand(-1, batch_size, -1, -1)
                    .reshape(-1, total_prompt_len, self.emb_dim),
                    input_emb.unsqueeze(0)
                    .expand(self.num_prompts, -1, -1, -1)
                    .reshape(-1, seq_len, self.emb_dim),
                ],
                dim=1,
            )  # [num_prompts*batch_size, total_prompt_len+seq_len, emb_dim]

            # Собираем маски
            expanded_prompt_mask = (
                prompt_mask.unsqueeze(1)
                .expand(-1, batch_size, -1)
                .reshape(-1, total_prompt_len)
            )
            if attention_mask is not None:
                expanded_input_mask = (
                    attention_mask.unsqueeze(0)
                    .expand(self.num_prompts, -1, -1)
                    .reshape(-1, seq_len)
                )
                combined_mask = torch.cat(
                    [expanded_prompt_mask, expanded_input_mask], dim=1
                )
            else:
                combined_mask = expanded_prompt_mask

            # Подаем в модель
            outputs = self.model(
                inputs_embeds=combined_emb.contiguous(),
                attention_mask=combined_mask.contiguous(),
                **kwargs,
            )

            # Переформатируем выходы
            logits = outputs.logits.view(
                self.num_prompts, batch_size, -1
            )

            # Вычисляем лоссы для каждого промпта
            if outputs.loss is not None:
                losses = outputs.loss.view(self.num_prompts, batch_size).mean(dim=1)
            else:
                losses = None

        else:
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
                        torch.ones(
                            prefix_len, dtype=torch.long, device=input_ids.device
                        ),
                        codebook_mask,
                        torch.ones(
                            suffix_len, dtype=torch.long, device=input_ids.device
                        ),
                    ]
                )

                prompt_mask_batch = current_prompt_mask.unsqueeze(0).expand(
                    batch_size, -1
                )

                if attention_mask is not None:
                    combined_mask = torch.cat(
                        [prompt_mask_batch, attention_mask], dim=1
                    )
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

        # Дополнительные лоссы
        aux_loss = (
            self.dissim_coef * dissim_loss + self.special_token_coef * special_penalty
        )

        return {"logits": logits, "losses": losses, "aux_loss": aux_loss}

    def get_codebook_tokens(self, no_gumbel=True, return_strings=True):
        """
        Возвращает промпты с чат-темплейтом в виде токенов
        """
        with torch.no_grad():
            _, prompt_indices, _ = self._quantize_codebook(training=not no_gumbel)

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
