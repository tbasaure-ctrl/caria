# PowerShell script to prepare repository for clean deployment
# This removes tracked data files and prepares for a fresh push

Write-Host "Preparing repository for clean deployment..." -ForegroundColor Cyan

# Step 1: Remove tracked data files from Git (but keep them locally)
Write-Host "`nStep 1: Removing tracked data files from Git..." -ForegroundColor Yellow

# Remove parquet files
Write-Host "  - Removing *.parquet files..." -ForegroundColor Gray
git rm --cached **/*.parquet 2>$null
git rm --cached caria_data/**/*.parquet 2>$null
git rm --cached data/**/*.parquet 2>$null

# Remove jsonl files
Write-Host "  - Removing *.jsonl files..." -ForegroundColor Gray
git rm --cached **/*.jsonl 2>$null
git rm --cached caria_data/**/*.jsonl 2>$null

# Remove csv files (if any are tracked)
Write-Host "  - Removing tracked *.csv files..." -ForegroundColor Gray
git rm --cached data/**/*.csv 2>$null
git rm --cached caria_data/data/**/*.csv 2>$null

# Remove data directories
Write-Host "  - Removing data/ directories..." -ForegroundColor Gray
git rm -r --cached data/ 2>$null
git rm -r --cached caria_data/data/ 2>$null
git rm -r --cached caria_data/silver/ 2>$null
git rm -r --cached silver/ 2>$null

Write-Host "[OK] Data files removed from Git tracking" -ForegroundColor Green

# Step 2: Show current status
Write-Host "`nStep 2: Current Git status..." -ForegroundColor Yellow
git status --short

# Step 3: Show what will be committed
Write-Host "`nStep 3: Files ready to commit..." -ForegroundColor Yellow
Write-Host "  Review the changes above" -ForegroundColor Gray
Write-Host "  Make sure .gitignore is correct" -ForegroundColor Gray
Write-Host "  Verify Dockerfile is ready" -ForegroundColor Gray

# Step 4: Instructions
Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Review the changes: git status" -ForegroundColor White
Write-Host "  2. Add files: git add ." -ForegroundColor White
Write-Host "  3. Commit: git commit -m 'Prepare for clean deployment'" -ForegroundColor White
Write-Host "  4. Push: git push origin main" -ForegroundColor White
Write-Host "  5. Deploy to Railway (backend)" -ForegroundColor White
Write-Host "  6. Deploy to Vercel (frontend)" -ForegroundColor White

Write-Host "`n[OK] Repository prepared! Follow the next steps above." -ForegroundColor Green


