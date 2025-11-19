# PowerShell script to create or update Google Cloud secrets
# This script is designed to be run in an environment where gcloud CLI is authenticated.

param (
    [string]$ProjectId = "caria-backend"
)

# Function to generate a random string for the JWT secret
function New-RandomString {
    param(
        [int]$length = 64
    )
    $bytes = New-Object byte[] $length
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    return [System.Convert]::ToBase64String($bytes).Substring(0, $length)
}

# Generate a secure random value for the JWT Secret Key
$jwtSecret = New-RandomString

# A dictionary of secrets to create.
$secrets = @{
    "reddit-client-id"      = "1eIYr0z6slzt62EXy1KQ6Q";
    "reddit-client-secret"  = "p53Yud4snfuadHAvgva_6vWkj0eXcw";
    "gemini-api-key"        = "AIzaSyC-EeIteUCY3gh0z4eFqRiwnqqkO9E5RQU";
    "fmp-api-key"           = "79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq";
    "postgres-password"     = "SimplePass123";
    "jwt-secret-key"        = $jwtSecret;
}

Write-Host "Starting to process secrets for project '$ProjectId'..."

foreach ($name in $secrets.Keys) {
    $value = $secrets[$name]
    Write-Host "Processing secret: '$name'..."

    # Check if the secret exists
    $secretExists = gcloud secrets describe $name --project=$ProjectId --quiet --error-format="none"
    
    if ($LASTEXITCODE -eq 0) {
        # Secret exists, add a new version
        Write-Host "Secret '$name' exists. Adding a new version."
        try {
            $value | gcloud secrets versions add $name --data-file=- --project=$ProjectId --quiet
            Write-Host "Successfully added new version to secret '$name'."
        } catch {
            Write-Error "Failed to add new version to secret '$name'. Error: $_"
        }
    } else {
        # Secret does not exist, create it
        Write-Host "Secret '$name' does not exist. Creating it."
        try {
            # Create the secret with the first version
            $value | gcloud secrets create $name --data-file=- --project=$ProjectId --replication-policy="automatic" --quiet
            Write-Host "Successfully created secret '$name'."
        } catch {
            Write-Error "Failed to create secret '$name'. Error: $_"
        }
    }
}

Write-Host "Secret processing complete."
