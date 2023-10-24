*The following commands reproduce the numbers in Table 4:*

**Cora:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=15 --constant_attention --dropout_edges --final_time=1.4962517538341513 --heads=1 --hidden_dim=40 --learning_rate=0.022070418257756394 --method=rk4 --num_epochs=66 --optimizer=adamax --pre_decoder_dropout=0.17117053951190142 --pre_encoder_dropout=0.32771015747073357 --use_squareplus --weight_decay=0.08225064156219447 --bracket hamiltonian --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --attention_ratio=3 --constant_attention --final_time=14.824043395883065 --heads=11 --hidden_dim=100 --learning_rate=0.01820535777750382 --method=rk4 --num_epochs=54 --optimizer=adamax --pre_decoder_dropout=0.25386151517370026 --pre_encoder_dropout=0.48085629022660525 --use_squareplus --weight_decay=0.023071665152910228 --bracket gradient_q_only --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=6 --dropout_edges --final_time=5.361505350658043 --heads=3 --hidden_dim=20 --learning_rate=0.02440889024398491 --method=rk4 --num_epochs=43 --optimizer=adam --pre_decoder_dropout=0.8011228591924469 --pre_encoder_dropout=0.24758675208439265 --use_squareplus --weight_decay=0.013253068075975684 --bracket double --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=8 --constant_attention --dropout_edges --final_time=7.531070496231473 --heads=6 --hidden_dim=60 --learning_rate=0.03203118089363822 --method=rk4 --num_epochs=100 --optimizer=adam --pre_decoder_dropout=0.7264340463153169 --pre_encoder_dropout=0.325787879936815 --use_squareplus --weight_decay=0.05554687125520461 --bracket metriplectic --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

**Citeseer:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=5 --dropout_edges --final_time=1.6949406964265097 --heads=3 --hidden_dim=70 --learning_rate=0.018649423346881095 --method=dopri5 --no_symmetrize --num_epochs=72 --optimizer=adam --pre_decoder_dropout=0.8288846298978417 --pre_encoder_dropout=0.4362665775326407 --weight_decay=0.09600724343348935 --bracket hamiltonian --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --attention_ratio=11 --constant_attention --dropout_edges --final_time=14.075206134955419 --heads=7 --hidden_dim=130 --learning_rate=0.010836609329130614 --method=rk4 --no_symmetrize --num_epochs=74 --optimizer=adamax --pre_decoder_dropout=0.8914479437527406 --pre_encoder_dropout=0.8044817482133417 --bracket gradient_q_only --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=5 --final_time=2.5848886199831553 --heads=13 --hidden_dim=120 --learning_rate=0.03558838986387741 --method=dopri5 --num_epochs=90 --optimizer=adam --pre_decoder_dropout=0.0053064970564912555 --pre_encoder_dropout=0.4250384361697026 --weight_decay=0.08533520328927428 --bracket double --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --attention_ratio=13 --constant_attention --dropout_edges --final_time=16.464667540977857 --heads=8 --hidden_dim=130 --learning_rate=0.04204399084273549 --method=dopri5 --no_symmetrize --num_epochs=50 --optimizer=adamax --pre_decoder_dropout=0.6817377303390313 --pre_encoder_dropout=0.09418864315908683 --weight_decay=0.08326454480644015 --bracket metriplectic --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

**Pubmed:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=1 --constant_attention --final_time=1.1685312848071372 --heads=2 --hidden_dim=70 --learning_rate=0.03342096023562196 --method=midpoint --num_epochs=93 --optimizer=adamax --pre_decoder_dropout=0.834039013509087 --pre_encoder_dropout=0.7286234822267508 --bracket hamiltonian --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=2 --constant_attention --dropout_edges --final_time=17.62508691508405 --heads=2 --hidden_dim=100 --learning_rate=0.04113579109263054 --method=midpoint --no_symmetrize --num_epochs=58 --optimizer=adam --pre_decoder_dropout=0.8959921323739093 --pre_encoder_dropout=0.6979978493733013 --use_squareplus --bracket gradient_q_only --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=3 --constant_attention --final_time=1.3724405367673833 --heads=6 --hidden_dim=130 --learning_rate=0.036887853613838166 --method=dopri5 --no_symmetrize --num_epochs=50 --optimizer=adam --pre_decoder_dropout=0.8406789491774697 --pre_encoder_dropout=0.4504030156079477 --use_squareplus --bracket double --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=14 --constant_attention --final_time=8.070156692916248 --heads=8 --hidden_dim=90 --learning_rate=0.04741295211398499 --method=rk4 --no_symmetrize --num_epochs=15 --optimizer=adam --pre_decoder_dropout=0.43580054085017483 --pre_encoder_dropout=0.4946924886095792 --use_squareplus --weight_decay=0.003412716428481233 --bracket metriplectic --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc --planetoid_split


*The following commands reproduce the numbers in Table 5:*

**Cora:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=15 --constant_attention --dropout_edges --final_time=1.4962517538341513 --heads=1 --hidden_dim=40 --learning_rate=0.022070418257756394 --method=rk4 --num_epochs=66 --optimizer=adamax --pre_decoder_dropout=0.17117053951190142 --pre_encoder_dropout=0.32771015747073357 --use_squareplus --weight_decay=0.08225064156219447 --bracket hamiltonian --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --attention_ratio=3 --constant_attention --final_time=14.824043395883065 --heads=11 --hidden_dim=100 --learning_rate=0.01820535777750382 --method=rk4 --num_epochs=54 --optimizer=adamax --pre_decoder_dropout=0.25386151517370026 --pre_encoder_dropout=0.48085629022660525 --use_squareplus --weight_decay=0.023071665152910228 --bracket gradient_q_only --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=6 --dropout_edges --final_time=5.361505350658043 --heads=3 --hidden_dim=20 --learning_rate=0.02440889024398491 --method=rk4 --num_epochs=43 --optimizer=adam --pre_decoder_dropout=0.8011228591924469 --pre_encoder_dropout=0.24758675208439265 --use_squareplus --weight_decay=0.013253068075975684 --bracket double --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=8 --constant_attention --dropout_edges --final_time=7.531070496231473 --heads=6 --hidden_dim=60 --learning_rate=0.03203118089363822 --method=rk4 --num_epochs=100 --optimizer=adam --pre_decoder_dropout=0.7264340463153169 --pre_encoder_dropout=0.325787879936815 --use_squareplus --weight_decay=0.05554687125520461 --bracket metriplectic --dataset Cora --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

**Citeseer:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=5 --dropout_edges --final_time=1.6949406964265097 --heads=3 --hidden_dim=70 --learning_rate=0.018649423346881095 --method=dopri5 --no_symmetrize --num_epochs=72 --optimizer=adam --pre_decoder_dropout=0.8288846298978417 --pre_encoder_dropout=0.4362665775326407 --weight_decay=0.09600724343348935 --bracket hamiltonian --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --attention_ratio=11 --constant_attention --dropout_edges --final_time=14.075206134955419 --heads=7 --hidden_dim=130 --learning_rate=0.010836609329130614 --method=rk4 --no_symmetrize --num_epochs=74 --optimizer=adamax --pre_decoder_dropout=0.8914479437527406 --pre_encoder_dropout=0.8044817482133417 --bracket gradient_q_only --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=5 --final_time=2.5848886199831553 --heads=13 --hidden_dim=120 --learning_rate=0.03558838986387741 --method=dopri5 --num_epochs=90 --optimizer=adam --pre_decoder_dropout=0.0053064970564912555 --pre_encoder_dropout=0.4250384361697026 --weight_decay=0.08533520328927428 --bracket double --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --attention_ratio=13 --constant_attention --dropout_edges --final_time=16.464667540977857 --heads=8 --hidden_dim=130 --learning_rate=0.04204399084273549 --method=dopri5 --no_symmetrize --num_epochs=50 --optimizer=adamax --pre_decoder_dropout=0.6817377303390313 --pre_encoder_dropout=0.09418864315908683 --weight_decay=0.08326454480644015 --bracket metriplectic --dataset Citeseer --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

**Pubmed:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=1 --constant_attention --final_time=1.1685312848071372 --heads=2 --hidden_dim=70 --learning_rate=0.03342096023562196 --method=midpoint --num_epochs=93 --optimizer=adamax --pre_decoder_dropout=0.834039013509087 --pre_encoder_dropout=0.7286234822267508 --bracket hamiltonian --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=2 --constant_attention --dropout_edges --final_time=17.62508691508405 --heads=2 --hidden_dim=100 --learning_rate=0.04113579109263054 --method=midpoint --no_symmetrize --num_epochs=58 --optimizer=adam --pre_decoder_dropout=0.8959921323739093 --pre_encoder_dropout=0.6979978493733013 --use_squareplus --bracket gradient_q_only --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=3 --constant_attention --final_time=1.3724405367673833 --heads=6 --hidden_dim=130 --learning_rate=0.036887853613838166 --method=dopri5 --no_symmetrize --num_epochs=50 --optimizer=adam --pre_decoder_dropout=0.8406789491774697 --pre_encoder_dropout=0.4504030156079477 --use_squareplus --bracket double --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=14 --constant_attention --final_time=8.070156692916248 --heads=8 --hidden_dim=90 --learning_rate=0.04741295211398499 --method=rk4 --no_symmetrize --num_epochs=15 --optimizer=adam --pre_decoder_dropout=0.43580054085017483 --pre_encoder_dropout=0.4946924886095792 --use_squareplus --weight_decay=0.003412716428481233 --bracket metriplectic --dataset Pubmed --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

**CoauthorCS:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=3 --attention_type=exp_kernel --constant_attention --dropout_edges --final_time=1.9297357175755492 --heads=6 --hidden_dim=120 --learning_rate=0.006927978420335585 --method=midpoint --num_epochs=60 --optimizer=adamax --pre_decoder_dropout=0.6023724948308988 --pre_encoder_dropout=0.7821816340479995 --weight_decay=0.030363097232124305 --bracket hamiltonian --dataset CoauthorCS --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=9 --attention_type=pearson --constant_attention --dropout_edges --final_time=12.89954446828914 --heads=5 --hidden_dim=50 --learning_rate=0.017295494195054568 --method=midpoint --no_symmetrize --num_epochs=37 --optimizer=adamax --pre_decoder_dropout=0.7705629234906949 --pre_encoder_dropout=0.8243526085477919 --weight_decay=0.002225105164363095 --bracket gradient_q_only --dataset CoauthorCS --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=15 --attention_type=pearson --constant_attention --dropout_edges --final_time=1.6662343471134777 --heads=11 --hidden_dim=120 --learning_rate=0.029296570855295214 --method=dopri5 --num_epochs=62 --optimizer=adamax --pre_decoder_dropout=0.8889422696847539 --pre_encoder_dropout=0.42456980040965414 --weight_decay=0.026485100606161428 --bracket double --dataset CoauthorCS --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

**Computers:**

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=4 --constant_attention --dropout_edges --final_time=1.122428413333182 --heads=7 --hidden_dim=70 --learning_rate=0.0052582667991976275 --method=rk4 --no_symmetrize --num_epochs=71 --optimizer=adam --pre_decoder_dropout=0.45672161996467586 --pre_encoder_dropout=0.7404846756498791 --bracket hamiltonian --dataset Computers --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --attention_ratio=6 --constant_attention --dropout_edges --final_time=14.104356697199368 --heads=3 --hidden_dim=20 --learning_rate=0.008838863722636222 --method=midpoint --no_symmetrize --num_epochs=70 --optimizer=adamax --pre_decoder_dropout=0.028052427263287372 --pre_encoder_dropout=0.15965471295583974 --use_squareplus --bracket gradient_q_only --dataset Computers --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --alpha_multiplier --attention_ratio=14 --constant_attention --dropout_edges --final_time=2.489290823108546 --heads=3 --hidden_dim=50 --learning_rate=0.002990761448012004 --method=dopri5 --no_symmetrize --num_epochs=96 --optimizer=adam --pre_decoder_dropout=0.32410734743901626 --pre_encoder_dropout=0.15427574138918365 --use_squareplus --bracket double --dataset Computers --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

**Photo:**

python random_initialization.py --alpha_multiplier --attention_ratio=2 --final_time=1.1643491931063448 --heads=3 --hidden_dim=30 --learning_rate=0.0050964052590728465 --method=midpoint --no_symmetrize --num_epochs=92 --optimizer=adam --pre_decoder_dropout=0.4481843701508042 --pre_encoder_dropout=0.4714884498407517 --use_squareplus --bracket hamiltonian --dataset Photo --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --attention_ratio=14 --constant_attention --final_time=16.67291807389978 --heads=15 --hidden_dim=80 --learning_rate=0.0031993828408762442 --method=dopri5 --no_symmetrize --num_epochs=97 --optimizer=adam --pre_decoder_dropout=0.6972272699755161 --pre_encoder_dropout=0.2584005485978887 --use_squareplus --bracket gradient_q_only --dataset Photo --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc

python random_initialization.py --add_self_loops --attention_ratio=9 --attention_type=pearson --dropout_edges --final_time=1.1845724707237246 --heads=8 --hidden_dim=150 --learning_rate=0.00659390170114268 --method=rk4 --no_symmetrize --num_epochs=65 --optimizer=adam --pre_decoder_dropout=0.7834049022086635 --pre_encoder_dropout=0.47249073519039525 --use_squareplus --weight_decay=0.0011426790329091775 --bracket double --dataset Photo --linear_encoder --linear_decoder --no_edge_encoder --no_edge_decoder --use_lcc