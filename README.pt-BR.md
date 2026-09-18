# nanollm — um motor de inferência de LLM otimizado

> **Idiomas:** [English](README.md) · **Português**

> Um motor de inferência feito do zero que roda o **GPT-2 de verdade** com as
> otimizações em cima das quais os motores de produção (TensorRT-LLM, vLLM,
> llama.cpp) são construídos: **KV-cache**, **atenção fundida estilo
> FlashAttention**, **quantização de pesos INT8/INT4** e **batching estático** —
> em Python/NumPy puro, com os kernels fundidos escritos em **Triton** para a GPU.
> Inferência eficiente de LLM é hoje o negócio central da NVIDIA; aqui está a
> maquinaria por baixo.

```bash
python scripts/download_model.py                       # baixa o GPT-2 (124M)
python scripts/generate.py "The future of AI is" --max-new 40
```
```
The future of AI is uncertain. The future of AI is uncertain. ...
                                     # GPT-2 124M real, gerado por este motor
```

---

## Resultados (CPU — o motor NumPy)

GPT-2 124M, `scripts/benchmark.py` + `scripts/eval_perplexity.py`:

| Otimização | Métrica | Resultado |
|---|---|---|
| **KV-cache** | throughput de decode | **10.0 → 39.1 tok/s (3.9× mais rápido)** |
| KV-cache | divisão de latência | TTFT 58 ms · por token 24 ms |
| **Batching estático** | throughput total | 38 → 50 → **53 tok/s** (batch 1→4→8) |
| **Quant INT8** | perplexidade / memória | **18.06 → 18.13** (+0.4%) · 498 → **243 MB** |
| Quant INT4 | perplexidade / memória | 18.06 → 31.38 · 498 → 201 MB |

![benchmark](results/benchmark.png)
![quantização](results/perplexity.png)

> O destaque: o **INT8 corta o modelo pela metade sem praticamente perder
> qualidade**, e o **KV-cache dá 3.9× de speedup no decode** — as duas
> otimizações que mais importam para servir modelos. Estes números são do motor
> NumPy na CPU; os **kernels Triton de GPU são medidos à parte numa Tesla T4** —
> ver *Resultados de GPU* abaixo.

---

## Resultados de GPU (kernels Triton, medidos numa Tesla T4 / Kaggle)

O `bench_gpu.py` exercita os dois kernels fundidos numa GPU de verdade: confere
cada um contra uma referência em PyTorch e então cronometra contra baselines. Os
kernels **compilam, rodam e estão numericamente corretos** em hardware real.

**FlashAttention (fundido, causal), fp32 — dimensões do GPT-2 124M (12 heads, head_dim 64):**

| seq len | triton (fundido) | PyTorch naive | torch SDPA | correção |
|--------:|-----------------:|--------------:|-----------:|----------|
| 256  | 0.353 ms | 0.310 ms | 0.210 ms | rel_err 1.2e-7 |
| 512  | 1.090 ms | 0.927 ms | 0.452 ms | rel_err 1.4e-7 |
| 1024 | 2.243 ms | 2.615 ms (**1.2× mais lento que o triton**) | 0.679 ms | rel_err 1.5e-7 |

**Dequant-matmul INT8 (fundido, GEMM da MLP 768→3072), ativações fp32:**

| M | triton (fundido) | dequant+mm sem fundir | fp32 cuBLAS | correção |
|--:|-----------------:|----------------------:|------------:|----------|
| 256  | 0.242 ms (**1.5× vs sem fundir**) | 0.366 ms | 0.272 ms | exato |
| 512  | 0.719 ms | 0.627 ms | 0.532 ms | exato |
| 1024 | 1.342 ms | 1.167 ms | 1.091 ms | rel_err 4.7e-7 |

Memória de peso: **fp32 → int8 é 4.0× menor**, mantido como int8 no caminho fundido.

> **Leitura honesta:** os kernels estão corretos em hardware real, e o caminho
> INT8 é um ganho de memória genuíno (4×). Em velocidade bruta eles são
> competitivos com os baselines naive, mas **não** batem o SDPA / cuBLAS do
> torch — que usam **Tensor Cores** em fp16, autotuning e pipelining de software
> que estes kernels feitos à mão em **fp32** ainda não têm (a T4 é sm_75; `tl.dot`
> em fp32 não aciona os tensor cores). Números reportados como medidos — vitórias
> e derrotas. Fechar essa lacuna (fp16 + autotune) é o primeiro item em *Próximos
> passos*.

---

## O que está implementado

| Recurso | Onde | Status |
|---|---|---|
| Carregar pesos open-source reais (GPT-2, safetensors) | [`loader.py`](nanollm/loader.py) | ✅ roda |
| Tokenizer BPE do GPT-2 do zero | [`tokenizer.py`](nanollm/tokenizer.py) | ✅ bate com ids canônicos |
| Forward do GPT-2 (NumPy), gera texto coerente | [`model.py`](nanollm/model.py) | ✅ roda |
| **KV-cache** no decode autoregressivo | [`model.py`](nanollm/model.py) | ✅ 3.9× |
| Atenção fundida **estilo FlashAttention** | [`attention.py`](nanollm/attention.py) (NumPy) · [`triton_kernels.py`](nanollm/triton_kernels.py) (GPU) | ✅ NumPy validado · ✅ Triton roda & correto na T4 |
| Quantização de pesos **INT8/INT4** | [`quant.py`](nanollm/quant.py) | ✅ roda |
| Kernel **dequant-matmul fundido** | [`triton_kernels.py`](nanollm/triton_kernels.py) | ✅ roda & correto na T4 (4× mem de peso) |
| **Batching estático** (left-pad + máscara) | [`generate.py`](nanollm/generate.py) | ✅ roda |
| Sampling (greedy / temperatura / top-k) | [`generate.py`](nanollm/generate.py) | ✅ roda |
| Métricas: tok/s, latência, memória, perplexidade | [`metrics.py`](nanollm/metrics.py), `scripts/` | ✅ roda |

O detalhamento do *porquê* de cada otimização está em
**[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)**.

---

## Início rápido

```bash
pip install -r requirements.txt              # numpy + safetensors
python scripts/download_model.py             # GPT-2 124M em models/gpt2/

python tests/test_nanollm.py                 # correção (tokenizer, flash==naive, KV-cache==sem-cache, quant, batching)
python scripts/generate.py "Once upon a time" --max-new 40 --temperature 0.8 --top-k 40
python scripts/generate.py "AI is" --quant int8         # pesos quantizados
python scripts/benchmark.py                  # tok/s, latência, memória  -> results/benchmark.csv
python scripts/eval_perplexity.py            # qualidade vs quantização  -> results/perplexity.csv
python scripts/plot_results.py               # -> results/*.png
python scripts/compare_baseline.py           # vs logits do HuggingFace (se transformers instalado)

# Só GPU (precisa de GPU NVIDIA + torch + triton):
python bench_gpu.py                          # kernels Triton: correção + timings vs PyTorch
```

---

## Arquitetura

```
nanollm/
├── tokenizer.py     BPE byte-level do GPT-2, do zero (vocab.json + merges.txt)
├── loader.py        baixa + lê safetensors -> {nome: np.float32}
├── config.py        GPT2Config (dims 124M / 355M / 774M / 1.5B)
├── model.py         forward do GPT-2 em NumPy + KVCache (blocos pre-LN, GELU, head amarrada)
├── attention.py     sdpa_naive  +  sdpa_flash (online softmax) — validados iguais
├── quant.py         quant simétrica por canal INT8/INT4 (+ empacotamento de bits INT4)
├── generate.py      sampling + loop de decode com KV-cache + batching estático (left-pad/máscara)
├── metrics.py       perplexidade, relatório de memória
└── triton_kernels.py  FlashAttention fundido + dequant-matmul INT8 fundido (Triton, GPU)
```

O modelo carrega os pesos como um `dict` de arrays NumPy; a quantização troca os
pesos lineares grandes por `QTensor`s, e o `_mm` do modelo despacha de forma
transparente (`x @ W` vs `qtensor.matmul(x)`). O backend de atenção é um switch de
uma linha (`attn_impl="naive"|"flash"`). Os kernels Triton em `triton_kernels.py`
são o caminho de GPU; o motor NumPy e os kernels de GPU compartilham a mesma
matemática, validada entre si na CPU e contra o PyTorch na GPU.

---

## Correção — como sabemos que está certo

`python tests/test_nanollm.py` (CPU):

```
[PASS] tokenizer roundtrip / ids canônicos       ( " the" -> [262], "Hello" -> [15496] )
[PASS] flash == naive  (prefill & decode)         rel_err ~1e-8
[PASS] erro de round-trip da quant INT8 / INT4    3.6e-3 / 6.4e-2
[PASS] KV-cache == sem-cache  (saída greedy idêntica)
[PASS] caminho flash-attn == caminho naive
[PASS] batch estático == solo  (idêntico por linha)
```

O `python bench_gpu.py` (GPU) confirma adicionalmente que os **kernels Triton**
batem com uma referência em PyTorch numa Tesla T4: FlashAttention com
`rel_err ~1e-7` vs `F.scaled_dot_product_attention`, e o dequant-matmul INT8 exato
vs `(x @ Wq.float()) * scale`.

E o `scripts/compare_baseline.py` confere os logits do nanollm contra o **GPT-2 do
HuggingFace** (se `transformers` estiver instalado) — batem até o arredondamento
de float32, confirmando que o forward feito do zero é o GPT-2 de verdade.

---

## Nota de honestidade (o que rodou onde)

O desenvolvimento foi feito numa máquina **só com CPU**, então os números do motor
NumPy acima (KV-cache, batching estático, perplexidade/memória da quantização) são
medições de CPU.

Os **kernels Triton de GPU** (`triton_kernels.py`) foram depois executados numa
**Tesla T4** (Kaggle) via `bench_gpu.py`: eles compilam, rodam e batem com a
referência em PyTorch (atenção `rel_err ~1e-7`; dequant-matmul INT8 exato) — a
seção *Resultados de GPU* reporta os timings medidos. Em velocidade bruta esses
kernels feitos à mão em **fp32** são competitivos com os baselines naive, mas
**não** batem o SDPA / cuBLAS do torch, que usam Tensor Cores em fp16 e autotuning
que os kernels ainda não têm. Todo número aqui é reportado exatamente como medido,
incluindo vitórias e derrotas. Não apresento números de GPU que não medi.

## Baselines & próximos passos

- **Baselines de correção:** GPT-2 do HuggingFace `transformers` (match de logits,
  `scripts/compare_baseline.py`); `F.scaled_dot_product_attention` do PyTorch e
  cuBLAS na GPU (`bench_gpu.py`).
- **Baselines de performance** (referências de produção que as otimizações
  almejam): `llama.cpp` (`llama-bench`) e `vLLM` (`api_server`).
- **Próximos passos:**
  - **`tl.dot` em fp16 + `@triton.autotune`** para que os kernels usem os Tensor
    Cores da T4 e blocos ajustados — o caminho direto para fechar a lacuna até
    SDPA/cuBLAS.
  - continuous batching + KV-cache paginado (PagedAttention),
  - quantizar também a embedding de tokens (leva o INT8 de ~2× para ~4×
    ponta-a-ponta),
  - GQA/RoPE para suportar Llama, e ativações fp16/bf16.

---

## Referências

- Radford et al. — *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Dao et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention*.
- Dettmers et al. — *LLM.int8()*; Frantar et al. — *GPTQ*.
- Kwon et al. — *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM).
- OpenAI Triton — tutorial de fused-attention.

---

*Código/comentários em inglês; README em inglês em [README.md](README.md).
Licença MIT. Companheiro de portfólio do [cudakit (kernels CUDA)](../cudakit) e
do [nabla (framework de autograd)](../nabla).*
