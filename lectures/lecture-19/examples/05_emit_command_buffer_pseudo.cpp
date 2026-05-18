// Pseudo C++ lowering helper for an educational NPU runtime backend.
#include <cstdint>
#include <vector>
#include <stdexcept>

struct PacketHeader {
  uint16_t kind;
  uint16_t sizeBytes;
  uint32_t flags;
  uint32_t waitToken;
  uint32_t signalToken;
  uint32_t debugTag;
};

enum : uint16_t {
  CMD_DMA_LOAD = 1,
  CMD_DMA_STORE = 2,
  CMD_MATMUL = 16,
  CMD_REQUANT = 17,
  CMD_FENCE = 241,
};

struct CommandBufferBuilder {
  std::vector<uint8_t> bytes;

  void align(unsigned alignment) {
    while (bytes.size() % alignment) bytes.push_back(0);
  }

  template <typename T>
  void append(const T &packet) {
    const auto *p = reinterpret_cast<const uint8_t *>(&packet);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }

  uint32_t dmaLoad(uint32_t srcDesc, uint32_t dstDesc, uint64_t bytesToCopy,
                   uint32_t wait, uint32_t signal);
  uint32_t matmul(uint32_t aDesc, uint32_t bDesc, uint32_t cDesc,
                  uint32_t tileM, uint32_t tileN, uint32_t tileK,
                  uint32_t wait, uint32_t signal);
  uint32_t fence(uint32_t wait);
};

// In a production backend, verifier checks must happen before serialization:
// - descriptor indices in range
// - alignment and packet size
// - token dependency graph
// - memory space legality
// - target capability support
