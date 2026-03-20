#include "text_decoder.h"
#include <cstdio>
#include <vector>
#include <map>

int main() {
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    const auto& cfg = decoder.get_config();
    printf("Model config:\n");
    printf("  hidden_size: %d\n", cfg.hidden_size);
    printf("  n_attention_heads: %d\n", cfg.n_attention_heads);
    printf("  n_key_value_heads: %d\n", cfg.n_key_value_heads);
    printf("  head_dim: %d\n", cfg.head_dim);
    printf("  rope_theta: %f\n", cfg.rope_theta);
    
    std::vector<int32_t> tokens = {151669, 151676, 151676, 151676, 151676, 
                                    151676, 151676, 151676, 151676, 151670};
    
    std::vector<float> logits;
    std::map<std::string, std::vector<float>> debug_tensors;
    
    if (!decoder.forward_debug(tokens.data(), tokens.size(), 0, logits, debug_tensors)) {
        fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    if (debug_tensors.count("debug_q0_rope")) {
        const auto& q_rope = debug_tensors["debug_q0_rope"];
        printf("\nC++ Q after RoPE (head 0, pos 0, first 10 values):\n");
        for (int i = 0; i < 10; ++i) {
            printf("%8.5f ", q_rope[i]);
        }
        printf("\n");
        
        printf("\nC++ Q after RoPE (head 0, pos 1, first 10 values):\n");
        for (int i = 0; i < 10; ++i) {
            printf("%8.5f ", q_rope[128 + i]);
        }
        printf("\n");
    }
    
    if (debug_tensors.count("debug_kq_scaled")) {
        const auto& kq = debug_tensors["debug_kq_scaled"];
        printf("\nC++ KQ scaled shape info: total size = %zu\n", kq.size());
        
        printf("\nC++ KQ scaled (head 0, 5x5):\n");
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int idx = i * 10 + j;
                printf("%8.4f ", kq[idx]);
            }
            printf("\n");
        }
        
        printf("\nC++ KQ scaled (head 1, 5x5):\n");
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int idx = 100 + i * 10 + j;
                printf("%8.4f ", kq[idx]);
            }
            printf("\n");
        }
    }
    
    return 0;
}
