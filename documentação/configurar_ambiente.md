# Documentação: `configurar_ambiente.sh`

O script `configurar_ambiente.sh` é uma ferramenta indispensável para automatizar a configuração inicial do ambiente de desenvolvimento. Ele cuida de todos os processos manuais que costumam causar erros entre diferentes máquinas, como a criação do ambiente virtual Python, a garantia de que o `pip` esteja atualizado e, o mais importante, a instalação inteligente e compatível com o hardware do sistema.

Uma das principais dores de cabeça no desenvolvimento com Deep Learning (utilizando o PyTorch) é garantir que os pacotes de áudio, visão e tensores estejam alinhados com os drivers da placa de vídeo. Este script resolve esse problema ao identificar se o computador utiliza hardware da AMD ou da NVIDIA / Intel. Isso permite que o projeto seja portável e otimizado sem que o desenvolvedor precise pesquisar comandos complexos no site oficial do PyTorch. No final de sua execução, o script também limpa conflitos em arquivos de texto de requisitos para que a instalação seja extremamente suave.

Abaixo, detalhamos como o script gerencia a instalação de dependências e os componentes do ambiente:

| Recurso | Função Principal | Detalhes do Script |
| :--- | :--- | :--- |
| `venv` | Isolamento do ambiente | Cria a pasta `./venv` com um interpretador Python 3 exclusivo. |
| `pip` | Gerenciador de pacotes | Atualiza o próprio instalador antes de baixar as bibliotecas. |
| `Hardware AMD` | Aceleração por ROCm | Instala versões do `torch` via repositório especializado (ROCm v6.0). |
| `Hardware NVIDIA` | Aceleração por CUDA | Instala via pacote de distribuição padrão do PyTorch. |
| `requirements.txt` | Dependências extras | Instala `transformers`, `flask`, `evaluate` e outras bibliotecas. |
| `temp_requirements` | Limpeza de conflitos | Remove referências ao `torch` no TXT para evitar regressão de versão. |

Portanto, o script elimina barreiras técnicas de entrada no projeto. Para utilizá-lo, o usuário só precisa conceder permissão de execução com `chmod +x configurar_ambiente.sh` e executá-lo em seguida. Após a finalização, basta ativar o ambiente virtual e todos os códigos de treinamento e teste estarão prontos para rodar com o máximo de performance que o hardware permitir.
