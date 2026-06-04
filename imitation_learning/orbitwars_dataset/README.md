How to create an Orbit Wars dataset:

1. Create `games.csv` from Meta Kaggle:

   ```bash
   python create_games_csv.py --competition_id <orbit_wars_competition_id>
   ```

2. Update `submissions.csv`:

   ```bash
   python update_submissions_csv.py
   ```

3. Download replays for a strong submission:

   ```bash
   python get_episodes.py --submission_id <submission_id>
   ```

4. Convert downloaded episodes into graph training shards:

   ```bash
   python convert_episodes.py --submission_id <submission_id> --num_workers 4
   ```

5. Train the graph policy:

   ```bash
   python ..\\orbitwars_train.py --submission_ids <submission_id>
   ```
