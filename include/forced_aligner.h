#pragma once

#include "qwen3asr_win_export.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qwen3_asr {

// Word with timestamp information
struct aligned_word {
    std::string word;
    float start;  // Start time in seconds
    float end;    // End time in seconds
};

// Alignment result
struct alignment_result {
    std::vector<aligned_word> words;
    bool success = false;
    std::string error_msg;
    
    // Timing info (in milliseconds)
    int64_t t_mel_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;
};

// ForcedAligner-specific hyperparameters
struct forced_aligner_hparams {
    // Audio encoder (LARGER than ASR)
    int32_t audio_encoder_layers = 24;
    int32_t audio_d_model = 1024;
    int32_t audio_attention_heads = 16;
    int32_t audio_ffn_dim = 4096;
    int32_t audio_num_mel_bins = 128;
    int32_t audio_conv_channels = 480;
    float audio_layer_norm_eps = 1e-5f;
    
    // Text decoder
    int32_t text_decoder_layers = 28;
    int32_t text_hidden_size = 1024;
    int32_t text_attention_heads = 16;
    int32_t text_kv_heads = 8;
    int32_t text_intermediate_size = 3072;
    int32_t text_head_dim = 128;
    float text_rms_norm_eps = 1e-6f;
    float text_rope_theta = 1000000.0f;
    int32_t vocab_size = 152064;
    
    // Classification head (instead of LM head)
    int32_t classify_num = 5000;
    
    // Special tokens
    int32_t timestamp_token_id = 151705;
    int32_t audio_start_token_id = 151669;
    int32_t audio_end_token_id = 151670;
    int32_t audio_pad_token_id = 151676;
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645;
    
    // Timestamp conversion
    int32_t timestamp_segment_time_ms = 80;  // Each class = 80ms
};

// Encoder layer for ForcedAligner audio encoder
struct fa_encoder_layer {
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_out_b = nullptr;
    
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    
    ggml_tensor * ffn_up_w = nullptr;
    ggml_tensor * ffn_up_b = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
    ggml_tensor * ffn_down_b = nullptr;
    
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
};

// Decoder layer for ForcedAligner text decoder
struct fa_decoder_layer {
    ggml_tensor * attn_norm = nullptr;
    
    ggml_tensor * attn_q = nullptr;
    ggml_tensor * attn_k = nullptr;
    ggml_tensor * attn_v = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
};

// ForcedAligner model
struct forced_aligner_model {
    forced_aligner_hparams hparams;
    
    // Audio encoder - Conv2D front-end
    ggml_tensor * conv2d1_w = nullptr;
    ggml_tensor * conv2d1_b = nullptr;
    ggml_tensor * conv2d2_w = nullptr;
    ggml_tensor * conv2d2_b = nullptr;
    ggml_tensor * conv2d3_w = nullptr;
    ggml_tensor * conv2d3_b = nullptr;
    ggml_tensor * conv_out_w = nullptr;
    
    // Audio encoder - Post-processing
    ggml_tensor * ln_post_w = nullptr;
    ggml_tensor * ln_post_b = nullptr;
    ggml_tensor * proj1_w = nullptr;
    ggml_tensor * proj1_b = nullptr;
    ggml_tensor * proj2_w = nullptr;
    ggml_tensor * proj2_b = nullptr;
    
    // Audio encoder layers
    std::vector<fa_encoder_layer> encoder_layers;
    
    // Text decoder - Embeddings
    ggml_tensor * token_embd = nullptr;
    
    // Text decoder layers
    std::vector<fa_decoder_layer> decoder_layers;
    
    // Text decoder - Final norm
    ggml_tensor * output_norm = nullptr;
    
    // Classification head (instead of LM head)
    ggml_tensor * classify_head_w = nullptr;
    ggml_tensor * classify_head_b = nullptr;
    
    // GGML context and buffers
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    // mmap state — must outlive all tensors backed by this mapping
    void * mmap_addr = nullptr;
    size_t mmap_size = 0;
    
    // Tensor name mapping
    std::map<std::string, ggml_tensor *> tensors;
    
    // Vocabulary
    std::vector<std::string> vocab;
    
    // BPE merge ranks: "first second" -> priority (lower = merge first)
    std::unordered_map<std::string, int> bpe_ranks;
    // Forward mapping: vocab token string -> token ID
    std::unordered_map<std::string, int32_t> token_to_id;
    
    // Korean dictionary for LTokenizer-style word splitting
    std::unordered_set<std::string> ko_dict;
};

// KV cache for decoder
struct fa_kv_cache {
    std::vector<ggml_tensor *> k_cache;
    std::vector<ggml_tensor *> v_cache;
    
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;
    int32_t n_used = 0;
    int32_t head_dim = 128;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;
};

// ForcedAligner state
struct forced_aligner_state {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    fa_kv_cache cache;
};

// ForcedAligner class
class ForcedAligner {
public:
    ForcedAligner();
    ~ForcedAligner();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);
    
    alignment_result align(const std::string & audio_path, const std::string & text,
                           const std::string & language = "");
    
    alignment_result align(const float * samples, int n_samples, const std::string & text,
                           const std::string & language = "");
    
    // Get error message
    [[nodiscard]] const std::string & get_error() const { return error_msg_; }
    
    // Check if model is loaded
    [[nodiscard]] bool is_loaded() const { return model_loaded_; }
    
    // Get hyperparameters
    [[nodiscard]] const forced_aligner_hparams & get_hparams() const { return model_.hparams; }
    
    std::vector<int32_t> tokenize_with_timestamps(const std::string & text,
                                                   std::vector<std::string> & words,
                                                   const std::string & language = "");
    
    bool load_korean_dict(const std::string & dict_path);
    
private:
    // Load model components
    bool parse_hparams(gguf_context * ctx);
    bool create_tensors(gguf_context * ctx);
    bool load_tensor_data(const std::string & path, gguf_context * ctx);
    bool load_vocab(gguf_context * ctx);
    
    // Initialize KV cache
    bool init_kv_cache(int32_t n_ctx);
    void clear_kv_cache();
    void free_kv_cache();
    
    // Audio encoding
    bool encode_audio(const float * mel_data, int n_mel, int n_frames,
                      std::vector<float> & output);
    
    // Build computation graph for decoder forward pass
    ggml_cgraph * build_decoder_graph(
        const int32_t * tokens, int32_t n_tokens,
        const float * audio_embd, int32_t n_audio,
        int32_t audio_start_pos);
    
    // Forward pass through decoder
    bool forward_decoder(
        const int32_t * tokens, int32_t n_tokens,
        const float * audio_embd, int32_t n_audio,
        int32_t audio_start_pos,
        std::vector<float> & output);
    
    // LIS-based timestamp correction (ported from HF fix_timestamp)
    std::vector<int32_t> fix_timestamp_classes(const std::vector<int32_t> & data);
    
    std::vector<float> classes_to_timestamps(const std::vector<int32_t> & classes);
    
    // Extract timestamp classes from logits
    std::vector<int32_t> extract_timestamp_classes(
        const std::vector<float> & logits,
        const std::vector<int32_t> & tokens,
        int32_t timestamp_token_id);
    
    // Build input sequence with audio placeholders
    std::vector<int32_t> build_input_tokens(
        const std::vector<int32_t> & text_tokens,
        int32_t n_audio_frames);
    
    // Find audio start position in token sequence
    int32_t find_audio_start_pos(const std::vector<int32_t> & tokens);
    
    // Model and state
    forced_aligner_model model_;
    forced_aligner_state state_;
    
    bool model_loaded_ = false;
    std::string error_msg_;
};

// Free model resources
void free_forced_aligner_model(forced_aligner_model & model);

std::vector<int32_t> simple_tokenize(const std::string & text,
                                      const std::vector<std::string> & vocab,
                                      std::vector<std::string> & words);

std::vector<std::string> tokenize_korean(const std::string & text,
                                          const std::unordered_set<std::string> & ko_dict);

} // namespace qwen3_asr
