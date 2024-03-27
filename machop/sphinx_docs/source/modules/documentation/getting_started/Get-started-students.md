# Additional Instructions for Students

Here are some additional tips that may help you in case you haven't worked with git/ssh before.

## SSH into an Imperial server

If you're using ``beholder1`` or ``ee-mill1``, first ssh into your required server as follows before following the setup instructions. Make sure you're running the Imperial VPN on your machine.

```bash
ssh <username>@<server_name>.ee.ic.ac.uk
```

## VS Code (optional)

My suggestion to work on the server is to install VS Code on your local machine and develop with it. You can also use your own IDE for the project, such as `emacs` or `vim`. 

1. Download and install [VS Code](https://code.visualstudio.com/)
2. Click the *extension* icon on the left bar (Cntrl + Shift + x), and search for *Remote - SSH* and *Remote Development* packages and install them.
3. Click *Help* at the top and choose *Show All Commands* (Cntrl + Shift + P) and open the console to run a few commands.
  - Type `ssh` in the console and choose *Remote SSH - Add New SSH Host*
  - Then you are asked to enter your ssh command, in our case, use: `ssh username@ee-beholder1.ee.ic.ac.uk`
  - Enter your password and you are in! (You may be asked to choose the platform for the first time, choose *Linux*).
4. Click *File* at the top-left and choose *Open Folder* to add the directory where you do your work to the explorer.
5. To use Terminal, you can click *Terminal* on the top and choose *New Terminal*.

Then you have a basic workspace!