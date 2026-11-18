************************************************************************************
MODULE NAME: RAKEIT

DESCRIPTION:
The RAKEIT module provides a compact one-way raking algorithm that can be plugged
into other production code. It assumes one variable being controlled to another. 
If the goal is a two-way rake, simply call the macro twice within a macro do loop. 
Inputs include the uncontrolled dataset and variable, the controlled dataset and variable, 
and the output dataset and variable (if different from the original), and a preferred number
of iterations. 

Author: 
Ben Bolender
US Census Bureau
Senior Advisor for Population Division
benjamin.c.bolender@census.gov

Last updated: 03/2021
************************************************************************************;

*Sample call:
%rakeit(
    UDSN=
    ,UVAR=
    ,CGRP=
    ,CDSN=
    ,CVAR=
    ,ODSN=
    ,OVAR=
    ,ITER=
    );
	
*USER INPUTS;
	*UDSN (required) = Starting dataset to be raked;
	*UVAR (required) = Uncontrolled variable to be controlled;
	*UGRP (optional) = Groups to control from if not already summed;
	*CGRP (required) = Control categories for summartion;
	*CVAR (required) = Control variable to rake to;
	*CDSN (optional) = Control data set name;
	*ODSN (optional) = Raked output dataset name;
	*OVAR (optional) = Raked output variable name;
	*ITER (optional) = Used to flag rakes in two way raking;

%macro rakeit(UDSN=,UVAR=,CGRP=,CDSN=,CVAR=,ODSN=,OVAR=,ITER=);
*If there is no control set, assume controlling to a different variable in the same set;
%if "&CDSN" = "" %then %let CDSN = &UDSN;
%if "&ODSN" = "" %then %let ODSN = &UDSN;
%if "&OVAR" = "" %then %let OVAR = &UVAR;
%if "&ITER" = "" %then %let ITER = 01;

*Read in input data;
/* %if "&UGRP" ^= "" %then %do;
	proc summary data = &USDN nway;
		class &UGRP;
		var &UVAR;
		output out = &UDSN; */
proc sort data = &UDSN presorted;
	by &CGRP;
	run;

*Read in control dataset or create it from another variable in the input data;
proc summary data = &CDSN nway;
	class &CGRP;
	var &CVAR;
	output out = control (drop=_:) sum=control;
	run;
proc sort data = control presorted;
	by &CGRP;
	run;

*Summarize input set to control groups;
proc summary data = &UDSN nway;
	class &CGRP;
	var &UVAR;
	output out = uncontrolled (drop=_:) sum=uncontrolled n=records;
	run;

*Merge controlled and uncontrolled data and calculate rake;
data rake;
	merge uncontrolled control;
	by &CGRP;
	if uncontrolled = . then uncontrolled = 0;
	if control = . then control = 0;
	if records > 0 then dist = control / records;
		else dist = 1;
	if uncontrolled > 0 then do;
		rake = control / uncontrolled;
		flag = 0;
		end;
		else do;
			rake = dist;
			flag = 1;
			end;
	keep &CGRP rake flag;
	run;

*Merge rake and uncontrolled data, apply rake;
data &ODSN;
	merge &UDSN rake;
	by &CGRP;
	if flag = 0 then &OVAR = &UVAR * rake;
		else &OVAR = rake;
	rake_&CDSN._&ITER = rake;
	/* flag_&CDSN._&ITER = flag; */
	drop rake flag;
	run;

%mend rakeit;