"""Run the existing three-model RD harness with high4-HEVC/low4-raw coding."""
from paper_bench import bench_kvcodec as benchmark
from paper_bench import hevc_token_hi4 as codec

benchmark.codec = codec


if __name__ == "__main__":
    benchmark.main()
