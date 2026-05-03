@echo off
echo ===================================================
echo Rebuilding Link Graph & Calculating PageRank
echo This process fetches pages to find links (Repair)
echo ===================================================
python manage.py rebuild-graph --db doner.db
pause
