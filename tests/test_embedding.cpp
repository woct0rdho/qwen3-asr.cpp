#include <cstdio>
#include <cstdlib>
#include <string>

#include "ggml.h"
#include "gguf.h"

int main() {
    const char* model_path = "models/qwen3-asr-0.6b-f16.gguf";
    
    ggml_context* ggml_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ggml_ctx,
    };
    
    gguf_context* ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to load GGUF\n");
        return 1;
    }
    
    ggml_tensor* embd = ggml_get_tensor(ggml_ctx, "token_embd.weight");
    if (!embd) {
        fprintf(stderr, "Failed to find token_embd.weight\n");
        gguf_free(ctx);
        return 1;
    }
    
    printf("Embedding tensor:\n");
    printf("  ne[0] = %lld (hidden_size)\n", (long long)embd->ne[0]);
    printf("  ne[1] = %lld (vocab_size)\n", (long long)embd->ne[1]);
    printf("  nb[0] = %zu (bytes per element)\n", embd->nb[0]);
    printf("  nb[1] = %zu (bytes per row/column)\n", embd->nb[1]);
    printf("  type = %d (1=F16)\n", embd->type);
    
    void* data = embd->data;
    
    int token_id = 151669;
    
    printf("\n=== Token %d embedding ===\n", token_id);
    printf("Reading as column (GGML convention):\n");
    for (int i = 0; i < 10; i++) {
        size_t elem_offset = i * embd->nb[0] + token_id * embd->nb[1];
        ggml_fp16_t* fp16_data = (ggml_fp16_t*)((char*)data + elem_offset);
        float val = ggml_fp16_to_fp32(*fp16_data);
        printf("  [%d] = %f\n", i, val);
    }
    
    printf("\nExpected (HuggingFace):\n");
    printf("  [0] = 0.01031494\n");
    printf("  [1] = -0.08007812\n");
    printf("  [2] = 0.02905273\n");
    printf("  [3] = 0.05322266\n");
    printf("  [4] = -0.00497437\n");
    
    printf("\n=== Checking ggml_get_rows behavior ===\n");
    printf("ggml_get_rows expects:\n");
    printf("  - src tensor with ne[0] = embedding_dim\n");
    printf("  - rows tensor with token indices\n");
    printf("  - Returns tensor with ne[0] = embedding_dim, ne[1] = n_tokens\n");
    printf("\nOur embedding tensor has ne[0]=%lld, ne[1]=%lld\n", 
           (long long)embd->ne[0], (long long)embd->ne[1]);
    printf("This means ggml_get_rows will extract 'rows' where each row is %lld elements\n",
           (long long)embd->ne[0]);
    printf("Row i is at offset i * nb[1] = i * %zu\n", embd->nb[1]);
    
    printf("\nFor token %d, row offset = %d * %zu = %zu\n", 
           token_id, token_id, embd->nb[1], (size_t)(token_id * embd->nb[1]));
    
    gguf_free(ctx);
    ggml_free(ggml_ctx);
    
    return 0;
}
