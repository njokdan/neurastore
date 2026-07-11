# Multi-stage build: compile in a full Rust image, ship only the binary
# in a minimal runtime image. The build stage is slow -- [profile.release]
# in Cargo.toml enables LTO and a single codegen unit, which is exactly
# what benefits the hot paths here (distance computation, graph
# traversal) but makes compilation itself take longer. That cost is paid
# once per image build, not per request, so it's a reasonable trade.

FROM rust:1-slim-bookworm AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./
COPY src ./src

RUN cargo build --release --bin server

# --- Runtime stage: no Rust toolchain, no source, just the binary ---
FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/server /app/server

# Data (WAL, SSTables) lives here -- mount a volume so it survives
# container restarts and recreation, not just process restarts.
VOLUME ["/data"]
EXPOSE 8080

ENTRYPOINT ["/app/server"]
CMD ["/data", "8080"]
