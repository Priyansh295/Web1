import React, { useState } from 'react';
import './lab1.css';

function Lab1() {
  const [activeStep, setActiveStep] = useState(null);

  const handleStepClick = (step) => {
    setActiveStep(activeStep === step ? null : step);
  };

  return (
    <div className="Lab1">
      <header>
        <h1>Lab Experiment 1: Setting Up AI Development Environment</h1>
      </header>

      <section>
        <h2>Step 1: Install Python</h2>
        <button className="step1" onClick={() => handleStepClick('step1')}>
          {activeStep === 'step1' ? 'Hide Details' : 'Show Details'}
        </button>
        {activeStep === 'step1' && (
          <ol className='open-sans-about-us-page'>
            <li>Visit the Python Official Website: Open your web browser and go to the official Python website. The URL is python.org.</li><br></br>
            <li>Navigate to the Downloads Section: On the Python website, you'll find a navigation bar at the top. Hover your cursor over the "Downloads" tab, and you'll see a dropdown menu.</li><br />
            <li>Choose Your Python Version: In the dropdown menu, you'll see different versions of Python available for download. Typically, you'll have two options: Python 3.x (the latest version) and Python 2.x (an older version, which is not recommended for new projects as it has reached its end-of-life). Click on the version you want to download. For most users, Python 3.x is the appropriate choice.</li><br />
            <li>Select the Installer: Once you've selected the version, you'll be directed to the downloads page for that version. You'll see various installers available for different operating systems (Windows, macOS, Linux, etc.). Choose the installer that corresponds to your operating system. For example, if you're using Windows, select the Windows installer.</li><br />
            <li>Download the Installer: Click on the download link for the installer, and your browser will start downloading the installer file. The file size may vary depending on your operating system and the version of Python you're downloading.</li> <br />
            <li>Run the Installer: Once the download is complete, locate the installer file on your computer (it's usually in your Downloads folder unless you specified a different location). Double-click the installer file to run it.</li> <br />
            <li>Install Python: The installer will launch a setup wizard. Follow the on-screen instructions to install Python on your computer. You can usually accept the default settings, but you may have the option to customize the installation (e.g., choosing the installation directory).</li> <br />
            <li>Check "Add Python to PATH" (Windows Only): On Windows, during the installation process, make sure to check the box that says "Add Python to PATH." This option ensures that Python is added to your system's PATH environment variable, allowing you to run Python from the command line more easily.</li> <br />
            <li>Complete the Installation: Once you've selected your preferences, click "Install" or "Finish" to complete the installation process. Python will be installed on your computer.</li> <br />
            <li>Verify the Installation: After the installation is complete, you can verify that Python has been installed correctly by opening a command prompt or terminal and typing python --version or python3 --version (depending on your operating system and configuration). This command should display the version of Python you've installed.</li> <br />
          </ol>
        )}
      </section>

      <section>
        <h2>Step 2: Install Visual Studio Code</h2>
        <button className="step2" onClick={() => handleStepClick('step2')}>
          {activeStep === 'step2' ? 'Hide Details' : 'Show Details'}
        </button>
        {activeStep === 'step2' && (
          <ol className='open-sans-about-us-page'>
            <li>Visit the Visual Studio Code Website: Open your web browser and go to the official Visual Studio Code website. The URL is code.visualstudio.com.</li> <br />
            <li>Download Visual Studio Code: On the homepage of the website, you'll see a prominent "Download" button. Click on it.</li> <br />
            <li>Select Your Operating System: Once you click the "Download" button, you'll be redirected to a page where you can choose the version of Visual Studio Code for your operating system. There are options available for Windows, macOS, and Linux. Click on the download link for the version that matches your operating system.</li> <br />
            <li>Download Begins: After selecting your operating system, the download should start automatically. If it doesn't, you might need to click on a specific link to initiate the download.</li> <br />
            <li>Locate the Installer: Once the download is complete, locate the installer file on your computer. By default, it's usually in your Downloads folder unless you specified a different location.</li> <br />
            <li>Run the Installer: Double-click on the installer file to run it. This will launch the setup wizard.</li> <br />
            <li>Install Visual Studio Code: Follow the instructions in the setup wizard to install Visual Studio Code on your computer. You can usually accept the default settings, but you may have the option to customize the installation (e.g., choosing the installation directory).</li> <br />
            <li>Launch Visual Studio Code: Once the installation is complete, you can launch Visual Studio Code by finding it in your list of installed applications or by searching for it in your computer's search bar.</li> <br />
            <li>Optional: Configure Settings: Upon launching Visual Studio Code for the first time, you might want to configure some settings according to your preferences. This includes choosing a color theme, configuring keyboard shortcuts, installing extensions, etc.</li> <br />
            <li>Start Coding: You're now ready to start coding! Visual Studio Code is a powerful code editor with features like syntax highlighting, code completion, debugging support, and more.</li> <br />
          </ol>
        )}
      </section>

      <section>
        <h2>Step 3: Additional Python Libraries</h2>
        <button className="step3" onClick={() => handleStepClick('step3')}>
          {activeStep === 'step3' ? 'Hide Details' : 'Show Details'}
        </button>
        {activeStep === 'step3' && (
          <ol className='open-sans-about-us-page'>
            <li>Open Command Prompt or Terminal: Depending on your operating system, open Command Prompt (Windows) or Terminal (macOS/Linux).</li> <br />
            <li>Check if pip is Installed: Type the following command and press Enter to check if pip, Python's package manager, is installed:</li> <br />
            <li>If pip is installed, you'll see its version number. If it's not installed, you'll need to install Python first, as pip comes with Python installations starting from Python 3.4.</li> <br />
            <li>Upgrade pip (Optional): It's a good practice to upgrade pip to the latest version. You can do this by running the following command:</li> <br />
            <li>Install numpy: Type the following command and press Enter to install the numpy library:</li> <br />
            <li>Install scipy: Next, install the scipy library by running the following command:</li> <br />
            <li>Install matplotlib: Install the matplotlib library for data visualization with the following command:</li> <br />
            <li>Install scikit-learn: Install scikit-learn, by running the following command</li> <br />
            <li>Install pandas: Finally, install the pandas library for data manipulation and analysis with the following command:</li> <br />
          </ol>
        )}
      </section>

      <footer>
        <p className="about-content">
          This lab experiment aims to familiarize students with the setup of an AI development environment, including the installation of necessary software and tools.
        </p>
      </footer>
    </div>
  );
}

export default Lab1;

