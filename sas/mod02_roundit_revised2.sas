************************************************************************************
MODULE NAME: ROUNDIT

DESCRIPTION:
The ROUNDIT module provides a compact one-way, greatest-mantissa rounding algorithm 
that can be plugged into other production code. It assumes one variable being 
controlled to another (possibly from the RAKEIT module). Inputs include the 
input dataset and variable, the list of categories within which to round (typically 
the categories by which the data was controlled), and the output dataset and variable 
(if different from the original). 

----------UPDATE 7/1/2026 and 7/23/2026----------
George updated this for Vintage 2026 to ensure that people cannot be stuffed into
out-of-universe GQ ages. This change is in response to a few cases that happened in 
V2025, one of which was a geocoding issue that required a moderate amount of stuffing
for a single county in Georgia. These cases can happen when controlling a 0 value in
the MARC file to a non-0 value in the subcounty file, since people need to be stuffed
somewhere for the totals to align. See PPCR #470 for V2025 and PPCR #481 for Vintage 2026. 
This update will force any stuffing within GQ Type to be within the appropriate 
age ranges, listed below. NOTE: the military universe was originally 17-65, which is 
what Decennial and the MARC file use, but at the CCB meeting on 7/23/2026 it was decided
to keep our 17-64 universe for Vintage 2026. So, I am making the military 17-64.

GQ Type 1: age must be 15-100 (0-14 is OOU)
GQ Type 2: age must be 0-25 (26+ is OOU)
GQ Type 3: age must be 20-100 (0-19 is OOU)
GQ Type 4: no age restrictions
GQ Type 5: age must be 16-65 (0-15 AND 66+ are OOU)
GQ Type 6: age must be 17-64 (0-16 AND 65+ are OOU) - changed 7/23/26, see note above
GQ Type 7: no age restrictions
----------UPDATE FOR 7/1/2026 and 7/23/2026----------

Author: 
Ben Bolender
US Census Bureau
Senior Advisor for Population Division
benjamin.c.bolender@census.gov

Last updated: 03/2021 -- By Ben
Last updated: 07/2026 -- By George M. Hayward AKA Merlin
************************************************************************************;

*Sample call: 
%roundit(
	IDSN=
	,IVAR=
	,LIST=
	,ODSN=
	,OVAR=
	);

*USER INPUTS;
	*IDSN (required) = Input (unrounded) dataset name;
	*IVAR (required) = Input (unrounded) variable name;
	*LIST (required) = Control groupings for stuffing;
	*ODSN (optional) = Output (rounded) dataset name;
	*OVAR (optional) = Output (rounded) variable name;

%macro roundit(IDSN=,IVAR=,LIST=,ODSN=,OVAR=);
*If no output dataset info is given, replace the original file and variable;
%if "&ODSN" = "" %then %let ODSN = &IDSN;
%if "&OVAR" = "" %then %let OVAR = &IVAR;

%let LAST = %qscan(&LIST,-1," ");

/* ******************************************************** */
/* First update for 2026: adding an age universe right away
to the input data set that is going to feed the rest of the 
macro. */

	/* ****************************************** */
	/* Creating check for age **and** GQT on file */
	/* ****************************************** */
	%let has_age = 0;
	%let has_gqt = 0;
	%let use_age_uni_rules = 0;

	%macro check_age_uni_vars;
	    %local dsid rc;

	    %let dsid = %sysfunc(open(&IDSN, i));

	    %if &dsid %then %do;

	        /* Check whether each variable exists */
	        %let has_age = %sysfunc(varnum(&dsid, age));
	        %let has_gqt = %sysfunc(varnum(&dsid, gqt));

	        %let rc = %sysfunc(close(&dsid));

	        /* Convert variable positions into simple 1/0 flags */
	        %if &has_age > 0 %then %let has_age = 1;
	        %else %let has_age = 0;

	        %if &has_gqt > 0 %then %let has_gqt = 1;
	        %else %let has_gqt = 0;

	        /* Apply age-universe rules only if BOTH variables exist */
	        %if &has_age = 1 and &has_gqt = 1 %then %let use_age_uni_rules = 1;
	        %else %let use_age_uni_rules = 0;

	    %end;
	    %else %do;
	        %let has_age = 0;
	        %let has_gqt = 0;
	        %let use_age_uni_rules = 0;
	    %end;

	%mend check_age_uni_vars;

	%check_age_uni_vars;

	/* ************************************************ */
	/* Creating age universe IF age **and** GQT on file */
	/* ************************************************ */

	data &IDSN;
	    set &IDSN;

	    age_uni = 1; /* Default: every row is eligible for stuffing */

	    %if &use_age_uni_rules = 1 %then %do;

	        /* Take rows out of universe only when AGE and TYPEVAR exist */
			if gqt = 1 and age le 14 then age_uni=0;
			if gqt = 2 and age ge 26 then age_uni=0;
			if gqt = 3 and age le 19 then age_uni=0;
			if gqt = 5 and age le 15 then age_uni=0;
			if gqt = 5 and age ge 66 then age_uni=0;
			if gqt = 6 and age le 16 then age_uni=0;
			if gqt = 6 and age ge 65 then age_uni=0; /* Changed from 66 to 65 on 7/23/26 */

	    %end;

	run;

/* End of first update */
/* ******************************************************** */

data rounding;
	set &IDSN;
	integer = floor(&IVAR);
	mantissa = &IVAR - integer;
	run;

proc sort data = rounding presorted;
	by &LIST;
	run;

proc summary data = rounding nway;
	class &LIST;
	var &IVAR integer;
	output out = stuffing (drop=_:) sum=total integer;
	run;

data stuffing;
	set stuffing;
	if total = . then total = 0;
	if integer = . then integer = 0;
	stuffpool = sum(round(total,1), -(integer));
	drop total integer;
	run;

data &ODSN;
	merge rounding stuffing;
	by &LIST;
	call streaminit(42);
	tiebreaker = rand('uniform');
	run;

proc sort data = &ODSN presorted;
	by &LIST descending mantissa descending integer descending tiebreaker;
	run;

data &ODSN;
	set &ODSN;
	by &LIST descending mantissa;

	/* Second update for V2026: this logic is updated to 
	only increment the counter for eligible rows (in universe)
	and to only stuff for eligible rows. Note that the old '<=
	stuffpool' is now '< stuffpool' because the original code increased 
	the counter first, then tested <=, while the new code tests <
	first, then increases the counter only when adding someone. */

	if first.&LAST then counter = 0;
	/* Add one only to eligible rows, without consuming
       the stuffing pool when age_uni = 0 */
    if age_uni = 1 and counter < stuffpool then do;
        counter + 1;
        &OVAR = integer + 1;
    end;

    else do;
        &OVAR = integer;
    end;

    drop integer mantissa tiebreaker age_uni;

	/* Note that we are also adding age_uni to the drop here since
	we don't need that anymore. End of second update. */

	run;

%mend roundit;
