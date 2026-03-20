#include <ggml.h>
#include <ggml-backend.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

static bool load_npy_f32(const char* path, std::vector<float>& data, int& rows, int& cols) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    
    char magic[6];
    fread(magic, 1, 6, f);
    
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    
    uint16_t header_len;
    fread(&header_len, 2, 1, f);
    
    std::vector<char> header(header_len + 1);
    fread(header.data(), 1, header_len, f);
    header[header_len] = '\0';
    
    // Parse shape from header (simplified)
    // Assuming shape is (rows, cols)
    char* shape_start = strstr(header.data(), "'shape': (");
    if (shape_start) {
        shape_start += 10;
        sscanf(shape_start, "%d, %d", &rows, &cols);
    }
    
    data.resize(rows * cols);
    fread(data.data(), sizeof(float), rows * cols, f);
    fclose(f);
    return true;
}

int main() {
    // Load mel spectrogram
    std::vector<float> mel_data;
    int n_mel, n_frames;
    if (!load_npy_f32("tests/reference/mel.npy", mel_data, n_mel, n_frames)) {
        printf("Failed to load mel.npy\n");
        return 1;
    }
    
    printf("Mel shape: [%d, %d]\n", n_mel, n_frames);
    printf("mel_data[0] = %.6f (should be mel[0,0])\n", mel_data[0]);
    printf("mel_data[1] = %.6f (should be mel[0,1])\n", mel_data[1]);
    printf("mel_data[%d] = %.6f (should be mel[1,0])\n", n_frames, mel_data[n_frames]);
    
    // Initialize GGML
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    
    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    // Create mel tensor with shape [n_frames, n_mel]
    struct ggml_tensor* mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, n_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);
    ggml_set_output(mel);
    
    printf("\nMel tensor shape: ne[0]=%lld, ne[1]=%lld\n", (long long)mel->ne[0], (long long)mel->ne[1]);
    
    // Build graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mel);
    
    // Allocate
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // Set mel data directly (no transpose)
    ggml_backend_tensor_set(mel, mel_data.data(), 0, n_mel * n_frames * sizeof(float));
    
    // Read back and verify
    std::vector<float> mel_read(n_mel * n_frames);
    ggml_backend_tensor_get(mel, mel_read.data(), 0, n_mel * n_frames * sizeof(float));
    
    printf("\nAfter loading into GGML tensor:\n");
    printf("mel_read[0] = %.6f (GGML[0,0])\n", mel_read[0]);
    printf("mel_read[1] = %.6f (GGML[1,0])\n", mel_read[1]);
    printf("mel_read[%d] = %.6f (GGML[0,1])\n", n_frames, mel_read[n_frames]);
    
    // GGML element [w, h] is at offset w + h * ne[0] = w + h * n_frames
    // So mel_read[0] = GGML[0, 0]
    // mel_read[1] = GGML[1, 0]
    // mel_read[n_frames] = GGML[0, 1]
    
    // We want GGML[w, h] = mel[h, w]
    // So GGML[0, 0] should be mel[0, 0] = mel_data[0]
    // GGML[1, 0] should be mel[0, 1] = mel_data[1]
    // GGML[0, 1] should be mel[1, 0] = mel_data[n_frames]
    
    printf("\nExpected:\n");
    printf("GGML[0,0] should be mel[0,0] = %.6f\n", mel_data[0]);
    printf("GGML[1,0] should be mel[0,1] = %.6f\n", mel_data[1]);
    printf("GGML[0,1] should be mel[1,0] = %.6f\n", mel_data[n_frames]);
    
    // Cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    
    return 0;
}
