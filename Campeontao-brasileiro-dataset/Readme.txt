Para a realização do TCC foi escolhido um dataset com dados do campeonato brasileiro de futebol de 2003 ate 2024,
este dataset tinha 4 repartições principais, sendo elas Estatísticas gerais da partida, os jogadores que tomaram cartão, os jogadores que fizeram gol e os dados da partida em geral. Para este trabalho apenas as tabelas de estatísticas e dos dados da partida serão utilizados. 

A Tabela de estatísticas contem os seguintes atributos: ID, Rodada, Clube, Chutes, Chutes no alvo, posse de bola, passes, precisão no passe, faltas, cartão amarelo, cartão vermelho, impedimentos e escanteios, desta tabela não foi excluído nenhum atributo.

A tabela de dados gerais da partida contem os seguintes atributos: ID, Rodada, date, hora, mandante, visitante, formação mandante, formação visitante, técnico mandante, vencedor, arena, mandante placar, visitante placar, mandante estado, visitante estado, desta tabela ficaram apenas os atributos de ID, Rodada, Hora, Mandante, Visitante, Mandante placar, Visitante Placar e Vencedor.

Também foram adicionados os atributos de ELO-Rating para cada equipe, sendo eles retirados do site https://www.yurimalheiros.com/elo-brasileirao/.

Após escolher os atributos que serão utilizados a tabela completa ficou da seguinte forma:
ID,rodada,data,hora,mandante,visitante,mandante_Placar,visitante_Placar,ELO_Rating_mandante,ELO_Rating_visitante,Chutes_mandante,Chutes_visitante,Chutes_no_alvo_mandante,Chutes_no_alvo_visitante,Posse_mandante,Posse_visitante,Passes_mandante,Passes_visitante,Precisão_passes_mandante,Precisão_passes_visitante,Faltas_mandante,Faltas_visitante,Cartão_amarelo_mandante,Cartão_amarelo_visitante,Cartão_vermelho_mandante,Cartão_vermelho_visitante,Impedimentos_mandante,Impedimentos_visitante,Escanteios_mandante,Escanteios_visitante,Vencedor. Os atributos da tabela de estatísticas foram dividas em mandante e visitante pois os dados estavam em tuplas separadas.



Depois de escolher os atributos foi realizado uma redução nos dados do dataset por falta de informação ou informação incompleta, sendo assim o dataset passou de conter dados de 2003 até 2024 para 2015 até 2023:

O corte de 2003 ate 2014 foi cortado por não conter informação das estatísticas da partidas;

Os anos de 2015 e 2016 não continham os dados de chute ao gol de cada equipe, para arrumar isso foi adicionado na mão 
os dados utilizando o site da ESPN como provedora dos dados.

O ano de 2024 foi cortado por não conter nenhuma informação das estáticas das partidas;

Em diversos anos do recorte que ira ser utilizado continham alguns jogos que não continham dados nenhum, por isso 
eles também foram cortados do dataset.



Algumas das tuplas continham um ou dois atributos faltando para corrigir isso, esses atributos foram generalizados:

Quando foi adicionado os dados para chutes no alvo nos anos de 2015 e 2016 houve algumas partidas que não foi possível adicionar o valor para essa estatística por no site da ESPN não conter esses doado mas especificamente, no ano de 2015 alguns jogos do figueirense e um jogo do cruzeiro e em 2016 alguns jogos do atlético-MG não continham nenhuma estática na ficha do jogo sendo assim não sendo possível adicionar esse valor, para que não ficassem vazias esse valor, foi colocado no lugar o placar do jogo como sendo o numero de chutes no alvo;

O atributo de precisão de passe em alguns jogos estava como NONE, para contornar isso foi adicionado no lugar a 
media geral da precisão de passe que é 80%;

O Atributo vencendo apenas estava o vencedor quando não tinha vencedor aparecia o seguinte símbolo - algo nada intuitivo, para corrigir isso o - foi substituído por Empate.
		
	