#include "text_decoder.h"

#include <cstdio>
#include <algorithm>

int main() {
    printf("=== Simple Decoder Test ===\n\n");
    
    qwen3_asr::TextDecoder decoder;
    
    printf("Loading model...\n");
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    const auto & cfg = decoder.get_config();
    printf("Model config:\n");
    printf("  vocab_size: %d\n", cfg.vocab_size);
    printf("  hidden_size: %d\n", cfg.hidden_size);
    printf("  n_decoder_layers: %d\n", cfg.n_decoder_layers);
    printf("  n_attention_heads: %d\n", cfg.n_attention_heads);
    printf("  n_key_value_heads: %d\n", cfg.n_key_value_heads);
    printf("  head_dim: %d\n", cfg.head_dim);
    printf("\n");
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    // Test tokens: "The capital of France is"
    // From Python: [785, 6722, 315, 9625, 374]
    std::vector<int32_t> input_tokens = {785, 6722, 315, 9625, 374};
    
    printf("Input tokens: ");
    for (auto t : input_tokens) printf("%d ", t);
    printf("\n\n");
    
    std::vector<float> logits;
    if (!decoder.forward(input_tokens.data(), input_tokens.size(), 0, logits)) {
        fprintf(stderr, "Forward failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    printf("Logits size: %zu (expected %d x %d = %d)\n", 
           logits.size(), (int)input_tokens.size(), cfg.vocab_size,
           (int)input_tokens.size() * cfg.vocab_size);
    
    printf("\nExpected token embeddings (from Python):\n");
    printf("  [-0.0119, 0.0179, -0.0371, 0.0204, 0.0087, -0.0021, 0.0066, 0.0183, -0.0444, 0.0439]\n");
    
    // Get logits for last token
    const float * last_logits = logits.data() + (input_tokens.size() - 1) * cfg.vocab_size;
    
    // Find top 5 predictions
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < cfg.vocab_size; ++i) {
        scores.push_back({last_logits[i], i});
    }
    std::sort(scores.begin(), scores.end(), [](auto & a, auto & b) {
        return a.first > b.first;
    });
    
    printf("\nTop 5 predictions for next token:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  %d. Token %d: %.4f\n", i+1, scores[i].second, scores[i].first);
    }
    
    // Expected: 12095 (Paris)
    printf("\nExpected token: 12095 (Paris)\n");
    printf("Got token: %d\n", scores[0].second);
    
    if (scores[0].second == 12095) {
        printf("\nTEST PASSED!\n");
        return 0;
    } else {
        printf("\nTEST FAILED!\n");
        return 1;
    }
}
