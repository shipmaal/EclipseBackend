# limb-data

Data-only branch (no shared history with `main`). `kernels/bootstrap.py --limb`
downloads these files from `raw.githubusercontent.com` at a pinned commit of
this branch when NAIF/PDS are unreachable (`--source mirror`), and checks each
against the SHA-256 pinned in `kernels/bootstrap.py` (also in `limb/SHA256SUMS`).

| file | origin |
| --- | --- |
| `limb/lola_ldem16_limb20.bin` | LRO LOLA `LDEM_16` (PDS `LRO-L-LOLA-3-RDR-V1`, public domain), cut to the lunar limb band by `kernels/limb_band.py` (byte-identical to a local cut) |
| `limb/moon_pa_de440_200625.bpc` | NAIF generic kernel `pck/moon_pa_de440_200625.bpc`, unmodified |
| `limb/moon_de440_250416.tf` | NAIF generic kernel `fk/satellites/moon_de440_250416.tf`, unmodified |

Never rewrite this branch: bootstrap pins a commit. Add new data as a new commit
and update the pin.
