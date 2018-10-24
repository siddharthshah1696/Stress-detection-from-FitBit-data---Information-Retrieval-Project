#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include<unistd.h> 

#include<fcntl.h> 


int main()
{
	 int fd[2];
	 pipe(fd);
	 int pid = fork();
	 if(pid>0)
	 {
	 	dup2(fd[1],1);
	 	char *cmd1={"/bin/ls -al"};
	 	system(cmd1);
	 	wait(NULL);
	 }
	 else
	 {
	 	dup2(fd[0],0);
	 	char *cmd2={"/usr/bin/tr a-z A-Z"};
	 	system(cmd2);
	 }
    
	return 0;
}