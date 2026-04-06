import os
import subprocess
import re
from pathlib import Path

class MekaWrapper:
    def __init__(self, meka_lib_path: str):
        self.meka_lib_path = os.path.abspath(meka_lib_path)

    def _parse_metrics(self, output: str):
        """Regex robusto para capturar F1 Micro e Macro."""
        metrics = {}
        # Captura variações de nome que o MEKA/MULAN podem usar
        patterns = {
            'micro_f1': r"Micro-average F(?:-measure|1)\s+([\d.]+)",
            'macro_f1': r"Macro-average F(?:-measure|1)\s+([\d.]+)"
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))
        return metrics

    def run_pipeline(self, arff_path, meka_classifier, num_labels, weka_classifier=None, meka_params=None, weka_params=None):
        # 1. Classpath (Linux)
        jars = [str(p) for p in Path(self.meka_lib_path).glob("*.jar")]
        classpath_full = ":".join(jars)

        # 2. Comando - Ordem Rigorosa e Uso do Separador '--'
        # [JAVA] [CP] [MEKA_CLASS] [EVAL_OPTIONS] [WRAPPER_OPTIONS] -- [INTERNAL_OPTIONS]
        command = [
            "java", "-Xmx4g", "-cp", classpath_full, 
            str(meka_classifier),
            "-t", os.path.abspath(arff_path),
            "-C", str(num_labels),
            "-x", "10"
        ]

        # Se houver parâmetros do wrapper (não usado no IBkk, mas bom manter)
        if meka_params:
            command.extend([str(p) for p in meka_params])

        # Se houver Base Learner (Weka)
        if weka_classifier:
            command.extend(["-W", str(weka_classifier), "--"])
            if weka_params:
                command.extend([str(p) for p in weka_params])
        # Se for um classificador direto (como o IBkk)
        elif weka_params:
            command.extend([str(p) for p in weka_params])

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            
            # Debug: Se falhar, agora veremos o comando EXATO com o --
            metrics = self._parse_metrics(result.stdout)
            
            if not metrics:
                print(f"\n⚠️  [DEBUG] Saída bruta do Java:\n{result.stdout}")
                print(f"⚠️  [DEBUG] Erro do Java (Stderr):\n{result.stderr}")
                print(f"⚠️  [DEBUG] Comando tentado:\n{' '.join(command)}")
            
            return metrics
        except Exception as e:
            print(f"❌ Erro fatal: {e}")
            return None