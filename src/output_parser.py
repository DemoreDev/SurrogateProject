import re

def parse_output(output_text: str) -> dict:
    # Varre a string de saída do Meka e extrai as métricas específicas do experimento

    if not output_text:
        return {
            "f1_real": None,
            "build_time_sec": None,
            "test_time_sec": None,
            "total_time_sec": None,
        }
    
    # Extraindo F1 (macro averaged by label)
    f1_match = re.search(r"F1 \(macro averaged by label\)\s+([0-9.]+)", output_text)
    f1_real = float(f1_match.group(1)) if f1_match else None

    # Extraindo os Tempos (Build, Test e Total)
    build_match = re.search(r"Build Time\s+([0-9.]+)", output_text)
    build_time = float(build_match.group(1)) if build_match else None
    
    test_match = re.search(r"Test Time\s+([0-9.]+)", output_text)
    test_time = float(test_match.group(1)) if test_match else None
    
    total_match = re.search(r"Total Time\s+([0-9.]+)", output_text)
    total_time = float(total_match.group(1)) if total_match else None

    return {
        "f1_real": f1_real,
        "build_time_sec": build_time,
        "test_time_sec": test_time,
        "total_time_sec": total_time,
    }