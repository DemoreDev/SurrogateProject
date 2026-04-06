import pandas as pd
import os

class DatasetExporter:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def export_to_arff(self, dfX, dfy, output_path):
        """
        Versão Definitiva: Labels no INÍCIO + Tipagem rigorosa.
        Isso alinha o arquivo com o parâmetro -C 19 positivo.
        """
        num_labels = dfy.shape[1]
        relation_name = f"'{self.dataset_name}: -C {num_labels}'"
        
        # 1. MUDANÇA CRUCIAL: Labels (dfy) vêm PRIMEIRO, Features (dfX) vêm DEPOIS
        # Também forçamos o tipo int para as labels aqui
        full_df = pd.concat([dfy.astype(int), dfX], axis=1)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"@relation {relation_name}\n\n")
            
            # 2. Definição de Atributos na ordem nova
            for col in full_df.columns:
                safe_col_name = str(col).replace("'", "\\'")
                
                # Verificamos se a coluna pertence ao set de labels
                if col in dfy.columns:
                    # Formato nominal exigido pelo MULAN
                    f.write(f"@attribute '{safe_col_name}' {{0,1}}\n")
                else:
                    f.write(f"@attribute '{safe_col_name}' numeric\n")
            
            f.write("\n@data\n")
            
            # 3. Escrita dos dados garantindo que labels sejam 0/1 sem .0
            # Usamos um gerador para economizar memória se o dataset for grande
            for row in full_df.itertuples(index=False):
                # Formata cada valor: se for label (inteiro), vira '0' ou '1'. 
                # Se for feature, mantém o float.
                formatted_row = []
                for i, val in enumerate(row):
                    # As primeiras 'num_labels' colunas são as labels
                    if i < num_labels:
                        formatted_row.append(str(int(val)))
                    else:
                        formatted_row.append(str(val))
                
                f.write(",".join(formatted_row) + "\n")
                
        print(f"✅ Arquivo exportado com as labels no INÍCIO: {output_path}")
        return output_path