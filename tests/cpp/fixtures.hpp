// Reader for the plain-text golden fixtures in tests/cpp/fixtures/.
//
// They were written by the retired Python oracle (tools/dump_oracle.py over the
// pure-Python app/ math, last present at commit fe1a64a) and are now FROZEN
// regression goldens: they pin today's results at the tolerances each test
// states. An intentional change of the arithmetic may move a value; the PR that
// does so re-baselines the affected records (from the core) and shows the
// reference-eclipse tests still hold -- a tolerance is never widened to hide a
// change nobody explained (CLAUDE.md, "Validation").
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fixtures {

inline const std::filesystem::path kDir = ECLIPSE_FIXTURE_DIR;
inline const std::filesystem::path kKernelDir = ECLIPSE_KERNEL_DIR;
inline const std::filesystem::path kMetakernel = kKernelDir / "eclipse.tm";

/// One whitespace-separated record; ``kind`` is the first token.
struct Record {
    std::string kind;
    std::vector<std::string> tokens;  // the rest
    // strtod, not std::stod: the latter throws out_of_range on a subnormal
    // such as 5e-324 (glibc sets ERANGE), which is a valid fixture value.
    double num(size_t i) const {
        const std::string& t = tokens.at(i);
        char* end = nullptr;
        const double v = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0') throw std::invalid_argument("not a number: " + t);
        return v;
    }
};

inline std::vector<Record> read(const std::string& name) {
    std::ifstream in(kDir / name);
    if (!in) throw std::runtime_error("fixture not found: " + (kDir / name).string());
    std::vector<Record> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Record r;
        ss >> r.kind;
        for (std::string t; ss >> t;) r.tokens.push_back(t);
        out.push_back(std::move(r));
    }
    return out;
}

/// The SPK the fixtures were dumped with (``spk`` record) is on disk.
inline bool pinned_spk_present(const std::vector<Record>& recs) {
    for (const auto& r : recs)
        if (r.kind == "spk") return std::filesystem::exists(kKernelDir / r.tokens.at(0));
    return false;
}

/// zlib's CRC-32 (reflected, polynomial 0xEDB88320), as Python's ``zlib.crc32``.
inline std::uint32_t crc32_file(const std::filesystem::path& path) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    std::ifstream in(path, std::ios::binary);
    std::uint32_t crc = 0xFFFFFFFFu;
    std::array<char, 1 << 16> buf;
    while (in.read(buf.data(), buf.size()) || in.gcount() > 0) {
        const auto n = static_cast<size_t>(in.gcount());
        for (size_t i = 0; i < n; ++i)
            crc = table[(crc ^ static_cast<unsigned char>(buf[i])) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/// The binary Earth PCK the fixtures were dumped with (``pck <name> <size>
/// <crc32>`` record) is byte-identical to the one on disk. NAIF regenerates
/// ``earth_latest_high_prec.bpc`` daily under the same name, so the ITRF93
/// rows are replayable only against the exact file; false when there is no
/// ``pck`` record (the dump had no binary PCK) or the file differs.
inline bool pinned_pck_matches(const std::vector<Record>& recs) {
    for (const auto& r : recs) {
        if (r.kind != "pck") continue;
        const auto path = kKernelDir / r.tokens.at(0);
        if (!std::filesystem::exists(path)) return false;
        if (std::filesystem::file_size(path) != std::stoull(r.tokens.at(1))) return false;
        char hex[9];
        std::snprintf(hex, sizeof hex, "%08x", crc32_file(path));
        return r.tokens.at(2) == hex;
    }
    return false;
}

}  // namespace fixtures
