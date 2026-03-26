Write-Host "🧹 Pulizia completa (con esclusioni protette)..." -ForegroundColor Cyan

# File da PROTEGGERE (non verranno mai cancellati)
$protectedFiles = @(
    "action_score.py",
    ".gitignore",
    "clean.ps1",
    "clean_auto.ps1",
    "clean_safe_complete.ps1",
    "README.md",
    "testing_model.py"
)

$removedCount = 0
$protectedCount = 0

# Funzione per controllare se un file è protetto
function Is-Protected($fileName) {
    foreach ($protected in $protectedFiles) {
        if ($fileName -eq $protected) {
            return $true
        }
    }
    return $false
}

# Pattern di file da rimuovere
$patterns = @(
    "MODELLO_*.h5",
    "MODELLO_*.keras",
    "*_memory.pkl",
    "*_epsilon.txt",
    "*_DQN*.png",
    "Test_*.png",
    "Score_*.png",
    "Train_*.png",
    "*success_rate*.csv",
    "*.csv"
)

$searchPaths = @(".", "baseline RL")

foreach ($path in $searchPaths) {
    if (-not (Test-Path $path)) { continue }
    
    foreach ($pattern in $patterns) {
        $files = Get-ChildItem -Path $path -Filter $pattern -File -ErrorAction SilentlyContinue
        
        foreach ($file in $files) {
            if (Is-Protected $file.Name) {
                Write-Host "  🛡️  Protetto: $($file.Name)" -ForegroundColor Blue
                $protectedCount++
                continue
            }
            
            Remove-Item $file.FullName -Force
            Write-Host "  ✅ Rimosso: $($file.FullName)" -ForegroundColor Green
            $removedCount++
        }
    }
}

Write-Host "`n📊 Riepilogo:" -ForegroundColor Cyan
Write-Host "  - File rimossi: $removedCount" -ForegroundColor Green
Write-Host "  - File protetti: $protectedCount" -ForegroundColor Blue
Write-Host "`nPulizia completata!" -ForegroundColor Cyan
