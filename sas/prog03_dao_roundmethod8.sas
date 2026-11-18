/* This file was created in D3 of V2026 in response to DAO 216-26: Disclosure Avoidance 
for Statistical Products. The DAO says "Coarsening shall be the preferred category 
of Disclosure Avoidance methods for all statistical products" and offers the following
definition: "Coarsening refers to the category of Disclosure Avoidance methods that 
involve reducing the level of detail or specificity of published statistics, such as 
through rounding, aggregating (grouping), and/or the use of ranges."

To accomplish this, we plan to coarsen (round) the base population estimates, and 
this program will likely present multiple alternatives as it evolves.

Began on 7/23/2026 by George M. Hayward and Luke T. Rogers. */ 

%macro check;
title "Check %upcase(&SYSLAST)  ###############################################";
proc means data = &SYSLAST n nmiss min max sum;
		run;
		title;
%mend check;

%include "/data/dec2k20/spenc357/CensusRakeAndRound/sas/mod01_rakeit.sas";
%include "/data/dec2k20/spenc357/CensusRakeAndRound/sas/mod02_roundit_revised2.sas";
/* Set libnames for output files */
	libname filein "/data/dec2k20/spenc357/CensusRakeAndRound/sas/";
	libname filemas "/data/dec2k20/spenc357/CensusRakeAndRound/sas/master";

data filein.PPMF_CELLS    ;
%let _EFIERR_ = 0; /* set the ERROR detection macro variable */
infile '/data/dec2k20/spenc357/CensusRakeAndRound/ppmf_cells.csv' delimiter = ',' MISSOVER DSD lrecl=32767 firstobs=2;
   informat nation $1. ;
   informat state $2. ;
   informat county $3. ;
   informat age best32. ;
   informat sex best32. ;
   informat cenrace best32. ;
   informat hisp best32. ;
   informat gqt best32. ;
   informat blend_pop best32. ;
   format nation $1. ;
   format state $2. ;
   format county $3. ;
   format age best12. ;
   format sex best12. ;
   format cenrace best12. ;
   format hisp best12. ;
   format gqt best12. ;
   format blend_pop best12. ;
input
            nation $
            state $
            county $
            age
            sex
            cenrace
            hisp
            gqt
            blend_pop
;
if _ERROR_ then call symputx('_EFIERR_',1);  /* set ERROR detection macro variable */
run;



/* ***************************************************************************** */
/* Loading county 31 race master file				                             */
/* ***************************************************************************** */

	/* Loading the pre-formatting version of our master file that I exported 
	for use here. */
		data rounded_input; 
		set filein.ppmf_cells; 
		run; 

/* ***************************************************************************** */
/* Some initial data prep and derivative files for raking                        */
/* ***************************************************************************** */

	/* Sorting */
		proc sort data=rounded_input;
		by nation state county age sex cenrace hisp gqt;
		run;

	/* Collapsing to GQT totals */
		proc summary data = rounded_input nway;
		class nation state county gqt;
		var blend_pop;
		output out=rounded_input_gqt (drop=_:) sum=;
		run;

			/* Creating a HH and GQ only file (if needed as controls) */

			/* HH file */
			data rounded_input_hh;
			set rounded_input_gqt;
			where gqt=0;
			drop gqt;
			run;

			/* GQ file */
			data rounded_input_gq;
			set rounded_input_gqt;
			where gqt > 0;
			drop gqt;
			run;

			/* Summing the GQ */
			proc summary data = rounded_input_gq nway;
			class nation state county;
			var blend_pop;
			output out=rounded_input_gq (drop=_:) sum=;
			run;

	/* Creating a county totals file (unrounded respop totals to use as control) */
		proc summary data=rounded_input_gqt nway;
		class nation state county;
		var blend_pop;
		output out=rounded_input_countytot (drop=_:) sum=;
		run;

	/* Creating a state totals by type file (unrounded totals to use as a control) */
		proc summary data=rounded_input_gqt nway;
		class nation state gqt;
		var blend_pop;
		output out=rounded_input_stategqt (drop=_:) sum=;
		run;

	/* Creating a county by GQT file that is ROUNDED to the nearest X, but don't
		allow anything to round to 0 (make it a 1 instead). This will be controlled
		to the unrounded state totals. */
		data rounded_input_gqt_r;
		set rounded_input_gqt;

		/* Rounding method */
		if blend_pop > 0 then do;
			blend_pop = round(blend_pop, 5);

			/* Prevent positive value from rounding to zero */
			if blend_pop = 0 then 
				blend_pop = 1;
		end;

		run;

	/* Raking the rounded county GQTs to the unrounded state GQTs, then rounding */
	%rakeit(
        UDSN=rounded_input_gqt_r
        ,UVAR=blend_pop
        ,CGRP=nation state gqt
        ,CDSN=rounded_input_stategqt
        ,CVAR=blend_pop
        ,ODSN=rounded_input_gqt_rc
        ,OVAR=blend_pop
        ,ITER=1
        );
	%check

	/* Temp check to see if it sums properly */
	/* Confirmed this works */
	proc summary data = rounded_input_gqt_rc nway;
	class nation state gqt;
	var blend_pop;
	output out=stategqt_check (drop=_:) sum=;
	run;
	
	%roundit(
        IDSN=rounded_input_gqt_rc
        ,IVAR=blend_pop
        ,LIST=nation state gqt
        ,ODSN=rounded_input_gqt_rcr
        ,OVAR=blend_pop
        );
	%check;

	/* Temp check to see if it sums properly */
	proc summary data = rounded_input_gqt_rcr nway;
	class nation state gqt;
	var blend_pop;
	output out=stategqt_check2 (drop=_:) sum=;
	run;

	/* Drop extra variables */
	data rounded_input_gqt_rcr; 
	set rounded_input_gqt_rcr; 
	keep nation state county gqt blend_pop; 
	run; 

	/* Sorting the data */
	proc sort data=rounded_input_gqt_rcr;
	by nation state county gqt;
	run;

/* ***************************************************************************** */
/* Rounding idea #6 - Round every cell to the nearest 5, but (1) leave zeros 
alone and (2) don't allow number to round to zero. Separately, create a file that 
rounds all the county-by-GQT totals to the nearest (X). Then, rake and round those
to make sure they sum up to the *unrounded* state-by-GQT totals. Then, rake the 
county detail file (rounded to nearest X) to those county-by-type totals. Then,
mantissa round/stuff. */

/* UPDATE FOR D5 IN V2026: Rounding the full detail file to multiples of 10 
	instead of 5 (which was D4) */
/* ***************************************************************************** */

	data base_round6;
	set rounded_input;

	/* Rounding method */
	if blend_pop > 0 then do;
		blend_pop = round(blend_pop, 5);

		/* Prevent positive value from rounding to zero */
		if blend_pop = 0 then 
			blend_pop = 1;

	end;

	run;

/* Raking to the rounded GQ totals by type, which sum to the state GQT totals */
	%rakeit(
        UDSN=base_round6
        ,UVAR=blend_pop
        ,CGRP=nation state county gqt
        ,CDSN=rounded_input_gqt_rcr
        ,CVAR=blend_pop
        ,ODSN=base_round6_c
        ,OVAR=blend_pop
        ,ITER=1
        );
	%check

	%roundit(
        IDSN=base_round6_c
        ,IVAR=blend_pop
        ,LIST=nation state county gqt
        ,ODSN=base_round6_cr
        ,OVAR=blend_pop
        );
	%check

/* Drop extra variables */
	data base_round6_cr; 
	set base_round6_cr; 
	keep nation state county age sex cenrace hisp gqt blend_pop; 
	run; 

	/* Sorting the data */
	proc sort data=base_round6_cr;
	by nation state county age sex cenrace hisp gqt;
	run;

	/* **************************************** */
	/* Review/checks on idea #6 				*/
	/* **************************************** */

	/* Collapsing to GQT totals */
		proc summary data = base_round6_cr nway;
		class nation state county gqt;
		var blend_pop;
		output out=base_round6_cr_gqt (drop=_:) sum=;
		run;

			/* Comparing to state GQT totals */
			proc summary data = base_round6_cr nway;
			class nation state gqt;
			var blend_pop;
			output out=base_round6_cr_stategqt (drop=_:) sum=;
			run;

	/* Comparing these totals to the original */
	/* NOTE: these are allowed to be different by TYPE but the county 
		TOTAL POP should be the same */
		proc compare base = rounded_input_gqt 
					 compare=base_round6_cr_gqt;
		id nation state county gqt;
		var blend_pop;
		run;

	/* Merging these two files together to create difference variables */
		data base_round6_diffs;
		merge rounded_input_gqt base_round6_cr_gqt (rename = blend_pop = blend_pop_r);
		by nation state county gqt;
		run;

	/* Creating differences variables */
		data base_round6_diffs;
		set base_round6_diffs;
		diff=blend_pop_r - blend_pop;
		absdiff=abs(diff);
		run;

	/* Some descriptive statistics on the differences */
		proc means data=base_round6_diffs sum N MEAN STD MIN MAX MODE MEDIAN P5 P95;
		var diff absdiff;
		run;

		/* For HH population */
		proc means data=base_round6_diffs sum N MEAN STD MIN MAX MODE MEDIAN P5 P95;
		where gqt = 0;
		var diff absdiff;
		run;

		/* For GQ population */
		proc means data=base_round6_diffs sum N MEAN STD MIN MAX MODE MEDIAN P5 P95;
		where gqt ne 0;
		var diff absdiff;
		run;

	/* ************************************** */
	/* Merging the full detail files together */
	/* ************************************** */
		data base_round6_diffs_fd;
		merge rounded_input base_round6_cr (rename = blend_pop = blend_pop_r);
		by nation state county age sex cenrace hisp gqt;
		run;

	/* Creating differences variables */
		data base_round6_diffs_fd;
		set base_round6_diffs_fd;
		diff=blend_pop_r - blend_pop;
		absdiff=abs(diff);
		run;

	/* Some descriptive statistics on the differences */
		proc means data=base_round6_diffs_fd sum N MEAN STD MIN MAX MODE MEDIAN P5 P95;
		var diff absdiff;
		run;

		/* For HH population */
		proc means data=base_round6_diffs_fd sum N MEAN STD MIN MAX MODE MEDIAN P5 P95;
		where gqt = 0;
		var diff absdiff;
		run;

		/* For GQ population */
		proc means data=base_round6_diffs_fd sum N MEAN STD MIN MAX MODE MEDIAN P5 P95;
		where gqt ne 0;
		var diff absdiff;
		run;

/* ***************************************************************************** */
/* Renaming the selected rounded file 'rounded' then using the exact same 
code from prog01 to transform, clean, and export the files, but this time with a 
suffix (_rounded) to indicate they are rounded versions of master files. This step 
is not strictly necessary, but it makes it explicit and works for now when I am 
testing various rounding methods in the same document. */
/* ***************************************************************************** */
	data rounded;
	set base_round6_cr;
	run;
	
	data filemas.rounded;
	set rounded;
	run;

/* ***************************************************************************** */
/* Creating new master files based on the rounded file							 */
/* ***************************************************************************** */

	/* Note that everything is GQT now instead of universe */
	data blend; set rounded;

		if gqt=1 then blend_gqtype1=blend_pop; 
		else blend_gqtype1=0; 

		if gqt=2 then blend_gqtype2=blend_pop;
		else blend_gqtype2=0; 
		 
		if gqt=3 then blend_gqtype3=blend_pop;
		else blend_gqtype3=0; 

		if gqt=4 then blend_gqtype4=blend_pop;
		else blend_gqtype4=0; 

		if gqt=5 then blend_gqtype5=blend_pop;
		else blend_gqtype5=0; 

		if gqt=6 then blend_gqtype6=blend_pop;
		else blend_gqtype6=0; 

		if gqt=7 then blend_gqtype7=blend_pop;
		else blend_gqtype7=0; 

		if gqt=0 then blend_hhpop=blend_pop; 
		else blend_hhpop=0; 

		blend_gqnoninst=sum(of blend_gqtype5-blend_gqtype7); 
		blend_gqinst=sum(of blend_gqtype1-blend_gqtype4); 
		blend_gqpop=blend_gqnoninst+blend_gqinst;

		blend_respop=blend_gqpop+blend_hhpop; 

		/* Creating race variable - need to update code elsewhere when running */
		if cenrace ge 6 then race6=6;
		else race6=cenrace; 
 
	run;  

	/* Sorting the data */
	proc sort data=blend;
	by nation state county age sex cenrace hisp;
	run;
		
	/* Remove extra blend variable from blend */
	data blend; set blend; 
	drop blend_pop; 
	run; 

/* ****************************************************************************** */
/* Using a proc sum to collapse the totals within blend_ variables                */
/* ****************************************************************************** */

	/* Need to cap age variable for county master file output. So, here we create 
	and then use "blend_85cap" for county and state, then use the full file (blend)
	for the nation */
		data blend_85cap;
		set blend;
		if age >85 then age=85;
		run;

		proc summary data = blend_85cap nway;
		class nation state county age sex cenrace race6 hisp gqt;
		var blend_:;
		output out=blend_85cap (drop=_:) sum=;
		run;
	
		/* ****************** */
		/* COUNTY             */
		/* ****************** */

		proc summary data = blend_85cap nway;
	    class nation state county age sex cenrace hisp;
	    var blend_:;
	    output out = filemas.bb2020_co31_rounded (drop=_:) sum=;
	    run;

			proc summary data = blend_85cap nway;
		    class nation state county age sex race6 hisp;
		    var blend_:;
		    output out = filemas.bb2020_co6_rounded (drop=_:) sum=;
		    run;

		/* ****************** */
		/* STATE              */
		/* ****************** */
		proc summary data = blend_85cap nway;
	    class nation state age sex cenrace hisp;
	    var blend_:;
	    output out = filemas.bb2020_st31_rounded (drop=_:) sum=;
	    run;

			proc summary data = blend_85cap nway;
		    class nation state age sex race6 hisp;
		    var blend_:;
		    output out = filemas.bb2020_st6_rounded (drop=_:) sum=;
		    run;

		/* ****************** */
		/* NATION             */
		/* ****************** */
		proc summary data = blend nway;
        class nation age sex cenrace hisp;
        var blend_:;
        output out = filemas.bb2020_nat31_rounded (drop=_:) sum=;
        run;

			proc summary data = blend nway;
	        class nation age sex race6 hisp;
	        var blend_:;
	        output out = filemas.bb2020_nat6_rounded (drop=_:) sum=;
	        run;
