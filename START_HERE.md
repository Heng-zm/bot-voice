# Start here

1. Extract this archive into your deployment directory.
2. Copy `.env.example` to `.env` and configure required secrets.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Preferred startup command:

   `python -m app.main`

The release also supports `python main.py` and `python app/main.py` for hosting
panels with fixed startup commands.

If the database tables do not exist yet, run `supabase_bot_setup.sql` in the
Supabase SQL editor before enabling persistent settings/admin features.
