// See eclipse/ephem.hpp. This translation unit is the only one that includes
// SpiceUsr.h / erfa.h.
#include "eclipse/ephem.hpp"

#include <mutex>
#include <type_traits>
#include <utility>

extern "C" {
#include "SpiceUsr.h"
#include "erfa.h"
#include "erfaextra.h"
}

namespace eclipse {

namespace {

// One lock around every CSPICE entry point: the kernel pool and the error
// subsystem are process-global state (roadmap §2).
std::mutex& spice_mutex() {
    static std::mutex m;
    return m;
}

// Put the toolkit in RETURN mode with no device output, exactly as spiceypy
// does at import (``erract("set", 10, "return")``, ``errdev("set", 10, "null")``).
// Idempotent; runs once under the lock on the first call.
void ensure_return_mode() {
    static bool done = false;
    if (done) return;
    // These two never fail in RETURN/ABORT mode with valid arguments.
    erract_c("SET", 0, const_cast<SpiceChar*>("RETURN"));
    errdev_c("SET", 0, const_cast<SpiceChar*>("NULL"));
    done = true;
}

// Message buffer sizes: the maxima documented in getmsg_c.html / qcktrc_c.html
// (SHORT 25, EXPLAIN 80, LONG 1840 chars) — the same values spiceypy uses.
constexpr SpiceInt kShortLen = 26;
constexpr SpiceInt kExplainLen = 101;
constexpr SpiceInt kLongLen = 1841;
constexpr SpiceInt kTraceLen = 1841;

[[noreturn]] void throw_spice_error() {
    SpiceChar short_msg[kShortLen], explain[kExplainLen], long_msg[kLongLen],
        trace[kTraceLen];
    getmsg_c("SHORT", kShortLen, short_msg);
    getmsg_c("EXPLAIN", kExplainLen, explain);
    getmsg_c("LONG", kLongLen, long_msg);
    qcktrc_c(kTraceLen, trace);
    reset_c();
    throw spice_error(short_msg, explain, long_msg, trace);
}

// Run ``f`` under the SPICE lock in RETURN mode and convert ``failed_c()`` into
// an exception — the C++ form of spiceypy's ``@spice_error_check``.
template <class F>
auto spice_call(F&& f) {
    std::scoped_lock lock(spice_mutex());
    ensure_return_mode();
    if constexpr (std::is_void_v<decltype(f())>) {
        f();
        if (failed_c()) throw_spice_error();
    } else {
        auto result = f();
        if (failed_c()) throw_spice_error();
        return result;
    }
}

std::string strip(std::string s) {
    const auto last = s.find_last_not_of(" \n");
    return last == std::string::npos ? std::string{} : s.substr(0, last + 1);
}

}  // namespace

spice_error::spice_error(std::string short_msg, std::string explain, std::string long_msg,
                         std::string traceback)
    : std::runtime_error(
          "\n================================================================================"
          "\n\nToolkit version: " + std::string(tkvrsn_c("TOOLKIT")) + "\n\n" + short_msg +
          " --\n" + strip(explain) + "\n" + strip(long_msg) + "\n\n" + strip(traceback) +
          "\n\n================================================================================"),
      short_(std::move(short_msg)),
      explain_(strip(std::move(explain))),
      long_(strip(std::move(long_msg))),
      traceback_(strip(std::move(traceback))) {}

namespace ephem {

std::string toolkit_version() {
    return spice_call([] { return std::string(tkvrsn_c("TOOLKIT")); });
}

std::string erfa_version() { return eraVersion(); }
std::string sofa_version() { return eraSofaVersion(); }

void furnish(std::string_view path) {
    const std::string p(path);
    spice_call([&] { furnsh_c(p.c_str()); });
}

void kclear() {
    spice_call([] { kclear_c(); });
}

int kernel_count(std::string_view kind) {
    const std::string k(kind);
    return spice_call([&] {
        SpiceInt n = 0;
        ktotal_c(k.c_str(), &n);
        return static_cast<int>(n);
    });
}

double str_to_et(std::string_view time_string) {
    const std::string s(time_string);
    return spice_call([&] {
        SpiceDouble et = 0.0;
        str2et_c(s.c_str(), &et);
        return et;
    });
}

std::string et_to_utc_iso(double et, int precision) {
    return spice_call([&] {
        // "ISOC" with 6 digits is 26 chars + NUL; leave room for any precision.
        SpiceChar buf[64];
        et2utc_c(et, "ISOC", static_cast<SpiceInt>(precision), sizeof buf, buf);
        return std::string(buf);
    });
}

Position body_position(std::string_view target, double et, std::string_view frame,
                       std::string_view abcorr, std::string_view observer) {
    const std::string t(target), f(frame), a(abcorr), o(observer);
    return spice_call([&] {
        Position p{};
        spkpos_c(t.c_str(), et, f.c_str(), a.c_str(), o.c_str(), p.km.data(), &p.light_time_s);
        return p;
    });
}

}  // namespace ephem
}  // namespace eclipse
