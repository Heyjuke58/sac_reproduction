# Notes

- Sampling:
    - am Anfang werden mit einer uniform policy steps gesampled
        - n_initial_exploration_steps: Wie viele Steps werden am Anfang mit uniform policy gemacht
        - wird gerundet auf vielfache von epoch_length
    - ab einem anderen Zeitpunkt wird angefangen zu trainieren (erst sollen genug samples im replay buffer sein)
        - min_replay_buffer_size: Wie viele Steps müssen im Replay buffer stehen, damit mit dem Training begonnen wird


# Fragen

- animationen in der präsentation? wie am besten?
- installation von mujoco erklären? welches OS?