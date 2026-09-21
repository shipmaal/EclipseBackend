// Reader for the plain-text oracle fixtures written by tools/dump_oracle.py.
#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
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
    double num(size_t i) const { return std::stod(tokens.at(i)); }
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

}  // namespace fixtures
