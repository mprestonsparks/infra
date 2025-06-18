#!/usr/bin/env pwsh
# Unified deployment validation script for infra + DEAN integration
# Validates both infra services and DEAN-specific requirements

param(
    [switch]$AutoFix = $false,
    [switch]$Verbose = $false,
    [string]$Environment = "development"
)

$ErrorActionPreference = "Stop"
$script:hasErrors = $false
$script:hasWarnings = $false

# Color output functions
function Write-Success { 
    param($Message)
    Write-Host "✓ $Message" -ForegroundColor Green 
}

function Write-Error { 
    param($Message)
    Write-Host "✗ $Message" -ForegroundColor Red
    $script:hasErrors = $true
}

function Write-Warning { 
    param($Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
    $script:hasWarnings = $true
}

function Write-Info { 
    param($Message)
    Write-Host "ℹ $Message" -ForegroundColor Cyan 
}

function Write-Header {
    param($Title)
    Write-Host "`n=== $Title ===" -ForegroundColor Magenta
}

# Check repository structure
function Test-RepositoryStructure {
    Write-Header "Checking Multi-Repository Structure"
    
    $requiredRepos = @{
        "DEAN" = @{
            Path = "../DEAN"
            Required = $true
            Description = "DEAN orchestration system"
        }
        "IndexAgent" = @{
            Path = "../IndexAgent"
            Required = $true
            Description = "Code indexing service"
        }
        "airflow-hub" = @{
            Path = "../airflow-hub"
            Required = $true
            Description = "Workflow orchestration"
        }
        "market-analysis" = @{
            Path = "../market-analysis"
            Required = $false
            Description = "Market analysis service"
        }
    }
    
    foreach ($repo in $requiredRepos.GetEnumerator()) {
        $repoName = $repo.Key
        $repoConfig = $repo.Value
        $repoPath = $repoConfig.Path
        
        if (Test-Path $repoPath) {
            Write-Success "$repoName repository found at $repoPath"
        } else {
            if ($repoConfig.Required) {
                Write-Error "$repoName repository not found at $repoPath - $($repoConfig.Description)"
            } else {
                Write-Warning "$repoName repository not found at $repoPath (optional)"
            }
        }
    }
}

# Check for BOM in all configuration files across repos
function Test-BOMInFiles {
    Write-Header "Checking for BOM characters in configuration files"
    
    $configPatterns = @(
        "*.yml",
        "*.yaml",
        "*.conf",
        "*.json",
        "*.env*",
        "docker-compose*.yml"
    )
    
    $searchDirs = @(".", "../DEAN", "../IndexAgent", "../airflow-hub")
    $bomFound = $false
    
    foreach ($dir in $searchDirs) {
        if (Test-Path $dir) {
            foreach ($pattern in $configPatterns) {
                $files = Get-ChildItem -Path $dir -Filter $pattern -Recurse -ErrorAction SilentlyContinue | 
                         Where-Object { $_.FullName -notmatch "node_modules|\.git|venv|__pycache__" }
                
                foreach ($file in $files) {
                    if (Test-Path $file.FullName) {
                        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
                        
                        # Check for UTF-8 BOM (EF BB BF)
                        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
                            Write-Error "BOM found in: $($file.FullName)"
                            $bomFound = $true
                            
                            if ($AutoFix) {
                                Write-Info "Removing BOM from $($file.Name)..."
                                $content = [System.IO.File]::ReadAllText($file.FullName)
                                [System.IO.File]::WriteAllText($file.FullName, $content, [System.Text.UTF8Encoding]::new($false))
                                Write-Success "BOM removed from $($file.Name)"
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (-not $bomFound) {
        Write-Success "No BOM characters found in configuration files"
    } elseif (-not $AutoFix) {
        Write-Warning "Run with -AutoFix to remove BOM characters automatically"
    }
}

# Validate environment variables for all services
function Test-EnvironmentVariables {
    Write-Header "Validating environment variables"
    
    # Check infra .env
    $infraEnvFile = ".env"
    $deanEnvFile = "../DEAN/.env"
    
    # Required variables for infra
    $infraRequiredVars = @{
        "CLAUDE_API_KEY" = @{
            MinLength = 10
            Description = "Claude API key"
        }
        "ANTHROPIC_API_KEY" = @{
            MinLength = 10
            Description = "Anthropic API key"
        }
        "DOCKER_DEFAULT_PLATFORM" = @{
            Description = "Docker platform (e.g., linux/amd64)"
        }
    }
    
    # Required variables for DEAN
    $deanRequiredVars = @{
        "JWT_SECRET_KEY" = @{
            MinLength = 32
            Description = "JWT signing key (min 32 chars)"
        }
        "POSTGRES_USER" = @{
            Description = "PostgreSQL username"
        }
        "POSTGRES_PASSWORD" = @{
            MinLength = 8
            Description = "PostgreSQL password (min 8 chars)"
        }
        "POSTGRES_DB" = @{
            Description = "PostgreSQL database name"
            ExpectedValue = "dean_production"
        }
        "REDIS_PASSWORD" = @{
            MinLength = 8
            Description = "Redis password (min 8 chars)"
        }
    }
    
    # Check infra environment
    if (Test-Path $infraEnvFile) {
        Write-Info "Checking infra environment variables..."
        Test-EnvFile -EnvFile $infraEnvFile -RequiredVars $infraRequiredVars
    } else {
        Write-Error "infra .env file not found"
        if ($AutoFix) {
            Write-Info "Creating .env from template..."
            if (Test-Path ".env.example") {
                Copy-Item ".env.example" ".env"
                Write-Success ".env created from template"
            }
        }
    }
    
    # Check DEAN environment if integrated
    if (Test-Path $deanEnvFile) {
        Write-Info "Checking DEAN environment variables..."
        Test-EnvFile -EnvFile $deanEnvFile -RequiredVars $deanRequiredVars
    }
}

function Test-EnvFile {
    param(
        $EnvFile,
        $RequiredVars
    )
    
    $envVars = @{}
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            $envVars[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    
    foreach ($var in $RequiredVars.GetEnumerator()) {
        $varName = $var.Key
        $varConfig = $var.Value
        $value = $envVars[$varName]
        
        if (-not $value) {
            Write-Error "$varName is not set in $EnvFile - $($varConfig.Description)"
        } else {
            # Check minimum length
            if ($varConfig.MinLength -and $value.Length -lt $varConfig.MinLength) {
                Write-Error "$varName is too short (${value.Length} chars, min $($varConfig.MinLength)) in $EnvFile"
            }
            
            # Check expected value
            if ($varConfig.ExpectedValue -and $value -ne $varConfig.ExpectedValue) {
                Write-Warning "$varName is '$value' but expected '$($varConfig.ExpectedValue)' in $EnvFile"
            }
            
            # Check for placeholder values
            if ($value -match "CHANGE_ME|REPLACE_ME|TODO|XXX") {
                Write-Error "$varName contains placeholder value in $EnvFile"
            }
        }
    }
}

# Check port conflicts across all services
function Test-PortAvailability {
    Write-Header "Checking port availability for all services"
    
    $allPorts = @{
        # Infra services
        5432 = "PostgreSQL"
        6379 = "Redis (infra)"
        6380 = "Redis (DEAN)"
        8080 = "Airflow"
        8081 = "IndexAgent"
        8090 = "DEAN Evolution API"
        8091 = "DEAN API"
        8092 = "DEAN Dashboard"
        8200 = "Vault"
        3000 = "Sourcebot/Grafana"
        6070 = "Zoekt"
        9090 = "Prometheus"
        
        # DEAN services (if standalone)
        80 = "HTTP (nginx)"
        443 = "HTTPS (nginx)"
        8082 = "DEAN Orchestrator"
    }
    
    foreach ($port in $allPorts.GetEnumerator()) {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        try {
            $tcpClient.Connect("localhost", $port.Key)
            $tcpClient.Close()
            Write-Warning "Port $($port.Key) is already in use ($($port.Value))"
        } catch {
            Write-Success "Port $($port.Key) is available ($($port.Value))"
        }
    }
}

# Check Docker environment
function Test-DockerEnvironment {
    Write-Header "Checking Docker environment"
    
    # Check if Docker is running
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker is running"
            
            # Check Docker version
            $dockerVersion = docker version --format '{{.Server.Version}}'
            Write-Info "Docker version: $dockerVersion"
        } else {
            Write-Error "Docker is not running"
            return
        }
    } catch {
        Write-Error "Docker command not found"
        return
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker Compose v2 is available"
        } else {
            # Try legacy docker-compose
            $legacyVersion = docker-compose --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Warning "Using legacy docker-compose (consider upgrading to Docker Compose v2)"
            } else {
                Write-Error "Docker Compose not found"
            }
        }
    } catch {
        Write-Error "Docker Compose not found"
    }
}

# Validate docker-compose files
function Test-DockerComposeFiles {
    Write-Header "Validating docker-compose files"
    
    $composeFiles = @(
        @{Path = "docker-compose.yml"; Name = "Infra main"},
        @{Path = "docker-compose.dean.yml"; Name = "DEAN integration"},
        @{Path = "docker-compose.dean.prod.yml"; Name = "DEAN production"},
        @{Path = "../DEAN/docker-compose.yml"; Name = "DEAN standalone"},
        @{Path = "../DEAN/docker-compose.prod.yml"; Name = "DEAN production"}
    )
    
    foreach ($file in $composeFiles) {
        if (Test-Path $file.Path) {
            Write-Info "Validating $($file.Name)..."
            # Check for valid YAML
            try {
                docker compose -f $file.Path config > $null 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "$($file.Name) is valid"
                } else {
                    Write-Error "$($file.Name) has syntax errors"
                }
            } catch {
                Write-Warning "Could not validate $($file.Name)"
            }
        }
    }
}

# Check for SSL certificates if DEAN nginx is used
function Test-SSLCertificates {
    Write-Header "Checking SSL certificates"
    
    $deanCertPath = "../DEAN/nginx/certs"
    
    if (Test-Path "../DEAN/docker-compose.prod.yml") {
        # Check if nginx service is defined
        $composeContent = Get-Content "../DEAN/docker-compose.prod.yml" -Raw
        if ($composeContent -match "nginx:") {
            Write-Info "DEAN nginx service detected, checking certificates..."
            
            if (-not (Test-Path $deanCertPath)) {
                Write-Error "Certificate directory $deanCertPath does not exist"
                if ($AutoFix) {
                    Write-Info "Creating certificate directory..."
                    New-Item -ItemType Directory -Path $deanCertPath -Force | Out-Null
                    Write-Success "Certificate directory created"
                }
            }
            
            $certFiles = @("server.crt", "server.key")
            $anyCertFound = $false
            
            foreach ($certFile in $certFiles) {
                $certPath = Join-Path $deanCertPath $certFile
                if (Test-Path $certPath) {
                    $anyCertFound = $true
                    Write-Success "Found certificate: $certFile"
                }
            }
            
            if (-not $anyCertFound) {
                Write-Warning "No SSL certificates found for DEAN nginx"
                Write-Info "Run ../DEAN/scripts/setup_ssl.ps1 to generate certificates"
            }
        }
    }
}

# Check service conflicts
function Test-ServiceConflicts {
    Write-Header "Checking for service conflicts"
    
    # Check for duplicate service names
    $allServices = @{}
    
    # Parse docker-compose files for service names
    $composeFiles = @(
        "docker-compose.yml",
        "docker-compose.dean.yml",
        "../DEAN/docker-compose.yml"
    )
    
    foreach ($file in $composeFiles) {
        if (Test-Path $file) {
            $content = Get-Content $file -Raw
            if ($content -match "services:") {
                # Simple regex to find service names
                $serviceMatches = [regex]::Matches($content, "^\s{2}(\w+):", [System.Text.RegularExpressions.RegexOptions]::Multiline)
                foreach ($match in $serviceMatches) {
                    $serviceName = $match.Groups[1].Value
                    if ($allServices.ContainsKey($serviceName)) {
                        Write-Warning "Duplicate service name '$serviceName' found in $file and $($allServices[$serviceName])"
                    } else {
                        $allServices[$serviceName] = $file
                    }
                }
            }
        }
    }
    
    # Check for database naming conflicts
    if ($allServices.ContainsKey("postgres") -and $allServices.ContainsKey("dean-postgres")) {
        Write-Warning "Both 'postgres' and 'dean-postgres' services exist - potential conflict"
        Write-Info "Consider using a single PostgreSQL instance with multiple databases"
    }
}

# Main validation function
function Start-ValidationChecks {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Multi-Repository Deployment Validation" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Environment: $Environment" -ForegroundColor White
    Write-Host "AutoFix: $(if ($AutoFix) { 'Enabled' } else { 'Disabled' })" -ForegroundColor White
    Write-Host "Time: $(Get-Date)" -ForegroundColor White
    
    # Run all checks
    Test-RepositoryStructure
    Test-DockerEnvironment
    Test-BOMInFiles
    Test-EnvironmentVariables
    Test-PortAvailability
    Test-DockerComposeFiles
    Test-SSLCertificates
    Test-ServiceConflicts
    
    # Summary
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Validation Summary" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    if ($script:hasErrors) {
        Write-Host "RESULT: FAILED - Errors found" -ForegroundColor Red
        if (-not $AutoFix) {
            Write-Host "`nRun with -AutoFix flag to attempt automatic fixes" -ForegroundColor Yellow
        }
        exit 1
    } elseif ($script:hasWarnings) {
        Write-Host "RESULT: PASSED WITH WARNINGS" -ForegroundColor Yellow
        Write-Host "The system can be deployed but review warnings above" -ForegroundColor Yellow
        exit 0
    } else {
        Write-Host "RESULT: PASSED - Ready for deployment!" -ForegroundColor Green
        exit 0
    }
}

# Run validation
Start-ValidationChecks