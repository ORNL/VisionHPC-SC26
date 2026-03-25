import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from colorama import init, Fore, Back, Style


if __name__ == "__main__":

    print(Fore.RED + "VisionHPC artifact:  Running artifact  ....")

    # Initialize colorama for cross-platform compatibility.
    # autoreset=True ensures the color is reset after each print statement.
    init(autoreset=True)

    #------Setup model
    
    model, tokenizer = FastVisionModel.from_pretrained(
            "../meta-llama/Llama-3.2-11B-Vision-Instruct",
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
            )
            
    model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = True, 
            finetune_language_layers   = True, 
            finetune_attention_modules = True,
            finetune_mlp_modules       = True,
            r = 16,           
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            random_state = 3443,
            use_rslora = False,
            loftq_config = None,
            )

    FastVisionModel.for_training(model)
    
    print(Fore.RED + "VisionHPC artifact:  Model set ....")

    #------Open images

    imageAXPY = Image.open("images/AXPY-image.png")
    imageSCAL = Image.open("images/SCAL-image.png")
    imageDOT = Image.open("images/DOT-image.png")
    imageASUM = Image.open("images/ASUM-image.png")
    imageNRM2 = Image.open("images/NRM2-image.png")
    imageGEMV = Image.open("images/GEMV-image.png")
    imageSYMV = Image.open("images/SYMV.png")
    imageGEMM = Image.open("images/GEMM.png")
    imageSYMM = Image.open("images/SYMM.png")
    imageStencil1D = Image.open("images/1D-Stencil-2point.png")
    imageStencil2D = Image.open("images/2D-Stencil-5point.png")
    imageLBM2DMacro = Image.open("images/2DLBM-Macro.png")
    imageLBM3DMacro = Image.open("images/3DLBM-Macro.png")

    print(Fore.RED + "VisionHPC artifact:  Images loaded ....")

    #------Load dataset: images and prompts (codes)

    dataset = [ 
            {'messages': [{'role': 'user',
                'content': [{'type': 'text',
                    'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: axpy(int N, float alpha, float *x, float *y)\n'},
                    {'type': 'image',
                        'image': imageAXPY}]},
                    {'role': 'assistant',
                        'content': [{'type': 'text',
                            'text':'\nvoid axpy(int N, float alpha, float *x, float *y) { \n  #pragma omp parallel for default(none) private(i) shared(alpha, N, x, y) \n  for (int i = 0; i < N; ++i) { \n    y[i] = alpha * x[i] + y[i]; \n  } \n}'
                            }]
                        }
                    ]}
            ,
            {'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: scal(int N, float alpha, float *x)\n'},
    {'type': 'image',
     'image': imageSCAL}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid scal(int N, float alpha, float *x) { \n  #pragma omp parallel for default(none) private(i) shared(alpha, N, x) \n  for (int i = 0; i < N; ++i) { \n    x[i] = alpha * x[i]; \n  } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: dot(int N, float *x, float *y)\n'},
    {'type': 'image',
     'image': imageDOT}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nfloat dot(int N, float *x, float *y) { \n float result = 0.0; \n #pragma omp parallel for reduction(+:result) default(none) private(i) shared(N, x, y) \n for (int i = 0; i < N; ++i) { \n  result += x[i] * y[i]; \n } \n return result; \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: asum(int N, float *x)\n'},
    {'type': 'image',
     'image': imageASUM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nfloat asum(int N, float *x) { \n float result = 0.0; \n #pragma omp parallel for reduction(+:result) default(none) private(i) shared(N, x) \n for (int i = 0; i < N; ++i) { \n  result += fabsf(x[i]); \n } \n return result; \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: asum(int N, float *x)\n'},
    {'type': 'image',
     'image': imageNRM2}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nfloat nrm2(int N, float *x) { \n float result = 0.0; \n #pragma omp parallel for reduction(+:result) default(none) private(i) shared(N, x) \n for (int i = 0; i < N; ++i) { \n  result += (x[i]*x[i]); \n } \n return sqrtf(result); \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: asum(int N, float *x)\n'},
    {'type': 'image',
     'image': imageGEMV}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid gemv(int M, int N, float alpha, float *a, float *x, float beta, float *y) { \n #pragma omp parallel for default(none) private(i, j) shared(M, N, alpha, a, x, beta, y) \n for (int i = 0; i < M; ++i) \n  y[i] = beta * y[i]; \n  for (int j = 0; j < N; ++j) { \n   y[i] += alpha * a[i * N + j] * x[j]); \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: asum(int N, float *x)\n'},
    {'type': 'image',
     'image': imageSYMV}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid symv(int M, int N, float alpha, float *a, float *x, float beta, float *y) { \n #pragma omp parallel for default(none) private(i, j) shared(M, N, alpha, a, x, beta, y) \n for (int i = 0; i < M; ++i) { \n  y[i] = beta * y[i]; \n  for (int j = 0; j <= i; ++j) { \n   y[i] += alpha * a[i * N + j] * x[j]); \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: asum(int N, float *x)\n'},
    {'type': 'image',
     'image': imageGEMM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid gemm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c) { \n #pragma omp parallel for default(none) private(i, j, m) shared(M, N, K, alpha, a, b, beta, c) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   c[i * N + j] = beta * c[i * N + j]; \n   for (int m = 0; m < K; ++m) { \n    c[i * N + j] += alpha * a[i * K + m] * b[m * N + j]; \n   } \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: asum(int N, float *x)\n'},
    {'type': 'image',
     'image': imageSYMM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c) { \n #pragma omp parallel for default(none) private(i, j, m) shared(M, N, K, alpha, a, b, beta, c) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   c[i * N + j] = beta * c[i * N + j]; \n   for (int m = 0; m <= i; ++m) { \n    c[i * N + j] += alpha * a[i * K + m] * b[m * N + j]; \n   } \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: stencil2D(int M, int N, float *u, float *v)\n'},
    {'type': 'image',
     'image': imageStencil2D}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid stencil2D(int M, int N, float *u, float *v) { \n #pragma omp parallel for default(none) private(i, j) shared(M, N, u, v) \n for (int i = 1; i < M - 1; ++i) { \n  for (int j = 1; j < N - 1; ++j) { \n   u[i * N + j] = 0.25 * (v[(i - 1) * N + j] + v[(i + 1) * N + j]) + 0.25 * (v[i * N + (j + 1)] + v[i * N + (j - 1)]) - v[i * N + j]; \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy)\n'},
    {'type': 'image',
     'image': imageLBM2DMacro}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy) { \n #pragma omp parallel for default(none) private(i, j, r) shared(M, N, Q, f1, u, v, p, cx, cy) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   for (int r = 0; r < Q; ++r) { \n    lp += f1[i * N * Q + j * Q + r]; \n   } \n   for (int r = 0; r < Q; ++r) { \n    lu += f1[i * N * Q + j * Q + r] * cx[r]; \n   } \n   for (int r = 0; r < Q; ++r) { \n    lv += f1[i * N * Q + j * Q + r] * cy[r]; \n   } \n   p[i * N + j] = lp; \n   u[i * N + j] = lu/lp; \n   v[i * N + j] = lv/lp; \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenMP code that implements the algorithm described in the image using this function name: 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy)\n'},
    {'type': 'image',
     'image': imageLBM3DMacro}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid 3dlbm_macro(int M, int N, int K, int Q, float *f1, float *u, float *v, float *w, float *p, int *cx, int *cy, int *cz) { \n #pragma omp parallel for default(none) private(i, j, l, r) shared(M, N, K, Q, f1, u, v, w p, cx, cy) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   \n  for (int l = 0; l < K; ++l) { \n    for (int r = 0; r < Q; ++r) { \n     lp += f1[i * N * K * Q + j * K * Q + l * Q + r]; \n    } \n    for (int r = 0; r < Q; ++r) { \n     lu += f1[i * N * K * Q + j * K * Q + l * Q + r] * cx[r]; \n    } \n    for (int r = 0; r < Q; ++r) { \n     lv += f1[i * N * K * Q + j * K * Q + l * Q + r] * cy[r]; \n    } \n    for (int r = 0; r < Q; ++r) { \n     lw += f1[i * N * K * Q + j * K * Q + l * Q + r] * cz[r]; \n    } \n    p[i * N * K + j * N + l] = lp; \n    u[i * N * K + j * K + l] = lu/lp; \n    v[i * N * K + j * K + l] = lv/lp; \n    w[i * N * K + j * K + l] = lw/lp; \n   } \n  } \n } \n}'
   }]
}
]}
,

{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: axpy(int N, float alpha, float *x, float *y)\n'},
    {'type': 'image',
     'image': imageAXPY}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid axpy(int N, float alpha, float *x, float *y) { \n  #pragma acc parallel loop independent present_or_copyin(x[0:N]) present_or_copyin(y[0:N]) present_or_copyout(y[0:N]) \n  for (int i = 0; i < N; ++i) { \n    y[i] = alpha * x[i] + y[i]; \n  }\n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: dot(int N, float *x, float *y)\n'},
    {'type': 'image',
     'image': imageDOT}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nfloat dot(int N, float *x, float *y) { \n float result = 0.0; \n #pragma acc parallel present_or_copyin(x[0:N]) present_or_copyin(y[0:N]) reduction(+:result) \n for (int i = 0; i < N; ++i) { \n  result += x[i] * y[i]; \n } \n return result; \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: dot(int N, float *x, float *y)\n'},
    {'type': 'image',
     'image': imageDOT}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nfloat dot(int N, float *x, float *y) { \n float result = 0.0; \n #pragma acc parallel present_or_copyin(x[0:N]) present_or_copyin(y[0:N]) reduction(+:result) \n for (int i = 0; i < N; ++i) { \n  result += x[i] * y[i]; \n } \n return result; \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: gemm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c)\n'},
    {'type': 'image',
     'image': imageGEMM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid gemm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c) { \n #pragma acc parallel loop independent present_or_copyin(a[0:M*K]) present_or_copyin(b[0:N*K]) present_or_copyin(c[0:M*N]) present_or_copyout(c[0:M*N]) collapse(2) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   c[i * N + j] = beta * c[i * n + j]; \n   for (int m = 0; m < K; ++m) { \n    c[i * N + j] += alpha * a[i * K + m] * b[m * N + j]; \n   } \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c)\n'},
    {'type': 'image',
     'image': imageSYMM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c) { \n #pragma acc parallel loop independent present_or_copyin(a[0:M*K]) present_or_copyin(b[0:N*K]) present_or_copyin(c[0:M*N]) present_or_copyout(c[0:M*N]) collapse(2) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   c[i * N + j] = beta * c[i * n + j]; \n   for (int m = 0; m < i; ++m) { \n    c[i * N + j] += alpha * a[i * K + m] * b[m * N + j]; \n   } \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: stencil1D(int N, float *u, float *v)\n'},
    {'type': 'image',
     'image': imageStencil1D}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid stencil1D(int N, float *u, float *v) { \n #pragma acc parallel loop independent present_or_copyin(u[0:M*N]) present_or_copyin(v[0:MN]) present_or_copyout(u[0:M*N]) \n for (int j = 1; j < N - 1; ++j) { \n   u[i] = 0.5 * (v[i - 1] + v[i + 1])) - v[i]; \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: stencil2D(int M, int N, float *u, float *v)\n'},
    {'type': 'image',
     'image': imageStencil2D}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid stencil2D(int M, int N, float *u, float *v) { \n #pragma acc parallel loop independent present_or_copyin(u[0:M*N]) present_or_copyin(v[0:MN]) present_or_copyout(u[0:M*N]) collapse(2) \n for (int i = 1; i < M - 1; ++i) { \n  for (int j = 1; j < N - 1; ++j) { \n   u[i * N + j] = 0.25 * (v[(i - 1) * N + j] + v[(i + 1) * N + j]) + 0.25 * (v[i * N + (j + 1)] + v[i * N + (j - 1)]) - v[i * N + j]; \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate an OpenACC code that implements the algorithm described in the image using this function name: 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy)\n'},
    {'type': 'image',
     'image': imageLBM2DMacro}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\nvoid 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy) { \n #pragma acc parallel loop independent present_or_copyin(f1[0:M*N*Q]) present_or_copyin(f2[0:M*N*Q]) present_or_copyin(u[0:M*N]) present_or_copyin(v[0:M*N]) present_or_copyin(p[0:M*N]) present_or_copyout(u[0:M*N]) present_or_copyout(v[0:M*N]) present_or_copyout(p[0:M*N]) collapse(2) \n for (int i = 0; i < M; ++i) { \n  for (int j = 0; j < N; ++j) { \n   for (int r = 0; r < Q; ++r) { \n    lp += f1[i * N * Q + j * Q + r]; \n   } \n   for (int r = 0; r < Q; ++r) { \n    lu += f1[i * N * Q + j * Q + r] * cx[r]; \n   } \n   for (int r = 0; r < Q; ++r) { \n    lv += f1[i * N * Q + j * Q + r] * cy[r]; \n   } \n   p[i * N + j] = lp; \n   u[i * N + j] = lu/lp; \n   v[i * N + j] = lv/lp; \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void axpy(int N, float alpha, float *x, float *y)\n'},
    {'type': 'image',
     'image': imageAXPY}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void axpy(int N, float alpha, float *x, float *y) { \n  int i = blockIdx.x * blockDim.x + threadIdx.x; \n if (i < N) { \n y[i] = alpha * x[i] + y[i]; \n }\n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void dot(int N, float *x, float *y, float *result)\n'},
    {'type': 'image',
     'image': imageDOT}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void dot(int N, float *x, float *y, float *result) { \n int ind = blockIdx.x * blockDim.x + threadIdx.x; \n if (ind == 0) { \n  for (int i = 0; i < N; ++i) { \n  result[0] += x[i] * y[i]; \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void dot(int N, float *x, float *y, float *result)\n'},
    {'type': 'image',
     'image': imageDOT}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void dot(int N, float *x, float *y, float *result) { \n int ind = blockIdx.x * blockDim.x + threadIdx.x; \n if (ind == 0) { \n  for (int i = 0; i < N; ++i) { \n  result[0] += x[i] * y[i]; \n  } \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void gemm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c)\n'},
    {'type': 'image',
     'image': imageGEMM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void gemm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c) { \n int i = blockIdx.x * blockDim.x + threadIdx.x; \n int j = blockIdx.y * blockDim.y + threadIdx.y; \n if (i < M && j < N) { \n  c[i * N + j] = beta * c[i * N + j]; \n  for (int m = 0; m < K; ++m) { \n   c[i * N + j] += alpha * a[i * K + m] * b[m * N + j]; \n  }\n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c)\n'},
    {'type': 'image',
     'image': imageSYMM}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'__global__ void symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c) { \n int i = blockIdx.x * blockDim.x + threadIdx.x; \n int j = blockIdx.y * blockDim.y + threadIdx.y; \n if (i < M && j < N) { \n  c[i * N + j] = beta * c[i * N + j]; \n  for (int m = 0; m < i; ++m) { \n   c[i * N + j] += alpha * a[i * K + m] * b[m * N + j]; \n  }\n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void stencil1D(int N, float *u, float *v)\n'},
    {'type': 'image',
     'image': imageStencil1D}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void stencil1D(int N, float *u, float *v) { \n int i = blockIdx.x * blockDim.x + threadIdx.x; \n if (i > 1 && i < N-1) { \n  u[i] = 0.5 * (v[i - 1] + v[i + 1])) - v[i]; \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void stencil2D(int M, int N, float *u, float *v)\n'},
    {'type': 'image',
     'image': imageStencil2D}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void stencil2D(int M, int N, float *u, float *v) { \n int i = blockIdx.x * blockDim.x + threadIdx.x; \n int j = blockIdx.y * blockDim.y + threadIdx.y; \n if (i > 1 && i < M-1 && j > 1 && j < N-1) { \n  u[i * N + j] = 0.25 * (v[(i - 1) * N + j] + v[(i + 1) * N + j]) + 0.25 * (v[i * N + (j + 1)] + v[i * N + (j - 1)]) - v[i * N + j]; \n } \n}'
   }]
}
]}
,
{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': '\nGenerate a CUDA code that implements the algorithm described in the image using this function name: __global__ void 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy)\n'},
    {'type': 'image',
     'image': imageLBM2DMacro}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text':'\n__global__ void 2dlbm_macro(int M, int N, int Q, float *f1, float *u, float *v, float *p, int *cx, int *cy) { \n int i = blockIdx.x * blockDim.x + threadIdx.x; \n int j = blockIdx.y * blockDim.y + threadIdx.y; \n if (i < M && j < N) { \n  for (int r = 0; r < Q; ++r) { \n   lp += f1[i * N * Q + j * Q + r]; \n  } \n  for (int r = 0; r < Q; ++r) { \n   lu += f1[i * N * Q + j * Q + r] * cx[r]; \n  } \n  for (int r = 0; r < Q; ++r) { \n   lv += f1[i * N * Q + j * Q + r] * cy[r]; \n  } \n  p[i * N + j] = lp; \n  u[i * N + j] = lu/lp; \n  v[i * N + j] = lv/lp; \n } \n}' 
   }]
}
]}
]

    print(Fore.RED + "VisionHPC artifact:  Data set (images and codes) for training (fine-tuning) loaded ....")

    #------Setup training (fine-tuning)
    
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=dataset,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                gradient_checkpointing=True,
                warmup_steps=5,
                max_steps=30,
                learning_rate=2e-4,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=5,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # For Weights and Biases
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=2048,
                ),
            )
    
    print(Fore.RED + "VisionHPC artifact:  Training (fine-tuning) parameters set ....")
    
    #------Train (fine-tuning)

    print(Fore.RED + "VisionHPC artifact:  Training (fine-tuning) base-model with VisionHPC dataset ....")
    
    trainer_stats = trainer.train()
    
    #------Activate Inference

    FastVisionModel.for_inference(model)
    
    #------Testing
    
    print(Fore.RED + "VisionHPC artifact:  Testing BLAS Level 1  ....")

    print(Fore.RED + "VisionHPC artifact:  Loading SCAL image (formula) ....")
    image = Image.open("images/SCAL-image.png")

    print(Fore.RED + "VisionHPC artifact:  Generating OpenMP SCAL code ....")
    
    messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Generate an OpenMP code that implements the algorithm described in the image using this function name: scal(int N, float alpha, float *x)"}
    ]}
    ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    
    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Generating CUDA/HIP SCAL code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate a CUDA code that implements the algorithm described in the image using this function name: __global__ void scal(int N, float alpha, float *x)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Testing BLAS Level 2  ....")
    
    print(Fore.RED + "VisionHPC artifact:  Loading SYMV image (formula) ....")
    
    image = Image.open("images/SYMV.png")

    print(Fore.RED + "VisionHPC artifact:  Generating OpenMP SYMV code ....")
    
    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate an OpenMP code that implements the algorithm described in the image using this function name: symv(int M, int N, float alpha, float *a, float *x, float beta, float *y)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Generating CUDA SYMV code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate a CUDA code that implements the algorithm described in the image using this function name: __global__ void symv(int M, int N, float alpha, float *a, float *x, float beta, float *y)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Testing BLAS Level 3  ....")
    
    print(Fore.RED + "VisionHPC artifact:  Loading SYMM image (formula) ....")
    
    image = Image.open("images/SYMM.png")
    
    print(Fore.RED + "VisionHPC artifact:  Generating OpenMP SYMM code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate an OpenMP code that implements the algorithm described in the image using this function name: symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Generating CUDA SYMM code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate a CUDA code that implements the algorithm described in the image using this function name: __global__ void symm(int M, int N, int K, float alpha, float *a, float *b, float beta, float *c)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Testing PDE (Euler) solvers ....")
    
    print(Fore.RED + "VisionHPC artifact:  Loading 2D (9 point) Euler image (formula) ....")
    
    image = Image.open("images/2D-Stencil-9point.png")
    
    print(Fore.RED + "VisionHPC artifact:  Generating OpenMP PDE code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate an OpenMP code that implements the algorithm described in the image using this function name: stencil2D(int M, int N, float *u, float *v)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Generating CUDA PDE code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate a CUDA code that implements the algorithm described in the image using this function name: __global__ void stencil2D(int M, int N, float *u, float *v)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Testing CFD (lattice-Boltzmann method [LBM]) solvers ....")
    
    print(Fore.RED + "VisionHPC artifact:  Loading 2D (collision) LBM  image (formula) ....")
    
    image = Image.open("images/2DLBM-Collision.png")
    
    print(Fore.RED + "VisionHPC artifact:  Generating OpenMP LBM code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate an OpenMP code that implements the algorithm described in the image using this function name: 2dlbm_coll(int M, int N, int Q, float *f1, float *f2, float *feq, float omega)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

    print(Fore.RED + "VisionHPC artifact:  Generating CUDA LBM code ....")

    messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Generate a CUDA code that implements the algorithm described in the image using this function name: __global__ void 2dlbm_coll(int M, int N, int Q, float *f1, float *f2, float *feq, float omega)"}
                ]}
            ]

    input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
            )

    inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
            ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=False) 
    _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=512,
            use_cache=False,
            temperature=0.0001,
            min_p=0.25
            )

