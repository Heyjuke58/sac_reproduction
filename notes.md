# Sachen die anders/besonders sind

- Sampling:
    - am Anfang werden mit einer uniform policy steps gesampled
        - n_initial_exploration_steps: Wie viele Steps werden am Anfang mit uniform policy gemacht
        - wird gerundet auf vielfache von epoch_length
    - ab einem anderen Zeitpunkt wird angefangen zu trainieren (erst sollen genug samples im replay buffer sein)
        - min_replay_buffer_size: Wie viele Steps müssen im Replay buffer stehen, damit mit dem Training begonnen wird
- im paper werden ko-varianzen erwähnt, aber... TODO
    - output size = 2 * action_space
- im policy loss wird im code nur eins der beiden value functions für KL benutzt
- policy loss hat auch regularization:
- reihenfolge der gradient updates ist im code anders
- die log_sigmas werden im code in `[-20, 2]` geclippt -> sigmas in `[2.1e-9, 7.4]`
- wir benutzen eine neuere Version von MuJoCo

- in v2: im Code wurde im neuesten Commit auf mean(Q1, Q2) anstatt min(Q1, Q2) gesetzt, und zwar nur im Policy update.
- in v2: policy benutzt softplus anstelle von clamping

# Logging
- Sollte Zeit (timestamps) und env steps enthalten um nach Zeit und Sampling Effizienz zu unterscheiden.
- Hyperparameter settings sollen am Anfang der csv auftauchen, muss beim Einlesen der Daten beachtet werden!
- Verschiedene runs werden in einer csv gespeichert und können über die Spalte 'seed' voneinander unterschieden werden.

# RUNS TODO:

Hauke:
3x TD3 Hopper 1e6 env steps ()
3x TD3 Cheetah 1,5e6 env steps (~4h each)

## Time for runs

- Konrad:
    - Cheetah * 3: 3.5h
    - Hopper * 3: 2.5h
    - need: SAC, SAC_V2, SAC_V2_FIXED_ALPHA * 4
    - SAC V1:
        - Cheetah, 1M env steps: 1.6h
        - Cheetah, 1.5M env steps: 2.5h
        - Hopper, 1M env steps: 1.8h
        - TODO: cheetah 1.5m x 3 + hopper 1m x 3 -> 12.9h
    - SAC V2 ohne alpha tuning
        - cheetah * 4 + hopper * 4 -> 16.2h
        - hopper * 4 -> 7.2h
    - SAC V2 mit alpha tuning
        - cheetah * 4 + hopper * 4 -> 16.2h
        - hopper * 4 -> 7.2h
    - SAC V1 mit hard target update
        - erstmal nicht
- Hauke:
    - TD3:
        - 3x TD3 Hopper 1M env steps (1.8h each) -> 5.5h done!
        - 3x TD3 Cheetah 1.5M env steps (~4h each) -> 12h