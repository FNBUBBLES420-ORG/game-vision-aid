# Instructions for installing Node.js
Write-Host "Please follow these steps to install Node.js:"

# Step 1: Download Node.js
Write-Host "1. Go to the Node.js download page: https://nodejs.org/"
Write-Host "2. Download the LTS (Long Term Support) version for Windows."

# Step 2: Install Node.js
Write-Host "3. Run the downloaded installer."
Write-Host "4. Follow the installation prompts and accept the default settings."

# Step 3: Verify the installation
Write-Host "5. Open a new PowerShell window."
Write-Host "6. Verify the installation by running the following commands:"
Write-Host "   - node -v"
Write-Host "   - npm -v"
Write-Host "   You should see the version numbers for Node.js and npm (Node Package Manager)."

# Step 4: Run your JavaScript file
Write-Host "7. Navigate to the directory where your JavaScript file is located using the 'cd' command."
Write-Host "8. Run your JavaScript file using Node.js by typing:"
Write-Host "   node your_script_name.js"
Write-Host "   Replace 'your_script_name.js' with the name of your JavaScript file."

# Pause to allow the user to read the instructions
Write-Host "Press any key to continue..."
[void][System.Console]::ReadKey($true)
