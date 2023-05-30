CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-11-08T10:09:32Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      �Tue May 30 16:58:31 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp126.r1i1p1f1.Amon.rsut.gn.v20210317/rsut_Amon_ACCESS-CM2_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp126_v20210317.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp126.r1i1p1f1.Amon.rsut.gn.v20210317/rsut_Amon_ACCESS-CM2_ssp126_r1i1p1f1_gn_210101-230012.yearmean.mul.areacella_ssp126_v20210317.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_ssp126.mergetime.nc
Tue May 30 16:58:31 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191108.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 05:52:53 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.fx.areacella.gn.v20191108/areacella_fx_ACCESS-CM2_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20191108.fldsum.nc
2019-11-08T10:09:32Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: CM2-historical; Local ID: bj594; Variable: rsut (['fld_s01i208'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0950-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   version       	v20191108      cmor_version      3.4.0      tracking_id       1hdl:21.14100/ae0d2880-10cc-4234-90ec-f56309c77f6a      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               H   	time_bnds                                 P   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               8   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               @   rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   cell_measures         area: areacella    history       u2019-11-08T10:09:31Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               `                Aq���   Aq��P   Aq�P   [0�/Aq�6�   Aq�P   Aq��P   [0�Aq���   Aq��P   Aq��P   [0��Aq��   Aq��P   Aq�dP   [1_sAq���   Aq�dP   Aq��P   [1)�Aq���   Aq��P   Aq�FP   [1�RAq�k�   Aq�FP   Aq��P   [0�Aq���   Aq��P   Aq�(P   [1/1Aq�M�   Aq�(P   Aq��P   [1U�Aq���   Aq��P   Aq�
P   [1$�Aq�/�   Aq�
P   Aq�{P   [19�Aq���   Aq�{P   Aq��P   [2��Aq��   Aq��P   Aq�]P   [2ӊAqĂ�   Aq�]P   Aq��P   [2}bAq���   Aq��P   Aq�?P   [1��Aq�d�   Aq�?P   Aq˰P   [1�~Aq���   Aq˰P   Aq�!P   [1�Aq�F�   Aq�!P   AqВP   [2�Aqз�   AqВP   Aq�P   [1b�Aq�(�   Aq�P   Aq�tP   [1�Aqՙ�   Aq�tP   Aq��P   [1w�Aq�
�   Aq��P   Aq�VP   [0�Aq�{�   Aq�VP   Aq��P   [1i2Aq���   Aq��P   Aq�8P   [2�Aq�]�   Aq�8P   Aq�P   [1��Aq���   Aq�P   Aq�P   [0�Aq�?�   Aq�P   Aq�P   [1/�Aq��   Aq�P   Aq��P   [0�uAq�!�   Aq��P   Aq�mP   [1�Aq��   Aq�mP   Aq��P   [1a�Aq��   Aq��P   Aq�OP   [2J�Aq�t�   Aq�OP   Aq��P   [1�_Aq���   Aq��P   Aq�1P   [1�Aq�V�   Aq�1P   Aq��P   [4�!Aq���   Aq��P   Aq�P   [8e�Aq�8�   Aq�P   Aq��P   [4ppAq���   Aq��P   Aq��P   [3E�Aq��   Aq��P   ArfP   [2\;Ar��   ArfP   Ar�P   [2\�Ar��   Ar�P   ArHP   [1�qArm�   ArHP   Ar�P   [1�Ar��   Ar�P   Ar*P   [3�tArO�   Ar*P   Ar�P   [2��Ar��   Ar�P   ArP   [2?�Ar1�   ArP   Ar}P   [2NpAr��   Ar}P   Ar�P   [1�zAr�   Ar�P   Ar_P   [1�Ar��   Ar_P   Ar�P   [1�Ar��   Ar�P   ArAP   [1��Arf�   ArAP   Ar�P   [1t�Ar��   Ar�P   Ar!#P   [1V�Ar!H�   Ar!#P   Ar#�P   [1��Ar#��   Ar#�P   Ar&P   [2pAr&*�   Ar&P   Ar(vP   [4�(Ar(��   Ar(vP   Ar*�P   [3�Ar+�   Ar*�P   Ar-XP   [2Z�Ar-}�   Ar-XP   Ar/�P   [2[�Ar/��   Ar/�P   Ar2:P   [2�Ar2_�   Ar2:P   Ar4�P   [2�Ar4��   Ar4�P   Ar7P   [2��Ar7A�   Ar7P   Ar9�P   [2 Ar9��   Ar9�P   Ar;�P   [2TjAr<#�   Ar;�P   Ar>oP   [4�}Ar>��   Ar>oP   Ar@�P   [4WArA�   Ar@�P   ArCQP   [3�ArCv�   ArCQP   ArE�P   [2��ArE��   ArE�P   ArH3P   [2�ArHX�   ArH3P   ArJ�P   [3.�ArJ��   ArJ�P   ArMP   [2u�ArM:�   ArMP   ArO�P   [2�iArO��   ArO�P   ArQ�P   [2�,ArR�   ArQ�P   ArThP   [2ہArT��   ArThP   ArV�P   [2��ArV��   ArV�P   ArYJP   [2�ArYo�   ArYJP   Ar[�P   [2ZvAr[��   Ar[�P   Ar^,P   [2[lAr^Q�   Ar^,P   Ar`�P   [2�MAr`��   Ar`�P   ArcP   [1�9Arc3�   ArcP   AreP   [1��Are��   AreP   Arg�P   [2�XArh�   Arg�P   ArjaP   [2��Arj��   ArjaP   Arl�P   [2b/Arl��   Arl�P   AroCP   [1��Aroh�   AroCP   Arq�P   [1\�Arq��   Arq�P   Art%P   [1wgArtJ�   Art%P   Arv�P   [1��Arv��   Arv�P   AryP   [2�Ary,�   AryP   Ar{xP   [28:Ar{��   Ar{xP   Ar}�P   [1��Ar~�   Ar}�P   Ar�ZP   [1��Ar��   Ar�ZP   Ar��P   [1��Ar���   Ar��P   Ar�<P   [1	qAr�a�   Ar�<P   Ar��P   [2��Ar���   Ar��P   Ar�P   [24�Ar�C�   Ar�P   Ar��P   [1��Ar���   Ar��P   Ar� P   [1.�Ar�%�   Ar� P   Ar�qP   [1߰Ar���   Ar�qP   Ar��P   [2p~Ar��   Ar��P   Ar�SP   [2�	Ar�x�   Ar�SP   Ar��P   [2��Ar���   Ar��P   Ar�5P   [2��Ar�Z�   Ar�5P   Ar��P   [2;�Ar���   Ar��P   Ar�P   [2B}Ar�<�   Ar�P   Ar��P   [1�Ar���   Ar��P   Ar��P   [2�Ar��   Ar��P   Ar�jP   [2�JAr���   Ar�jP   Ar��P   [2�Ar� �   Ar��P   Ar�LP   [2��Ar�q�   Ar�LP   Ar��P   [2�^Ar���   Ar��P   Ar�.P   [2m�Ar�S�   Ar�.P   Ar��P   [2�FAr���   Ar��P   Ar�P   [38�Ar�5�   Ar�P   Ar��P   [3�#Ar���   Ar��P   Ar��P   [6��Ar��   Ar��P   Ar�cP   [5�sAr���   Ar�cP   Ar��P   [55Ar���   Ar��P   Ar�EP   [3�'Ar�j�   Ar�EP   ArĶP   [4:Ar���   ArĶP   Ar�'P   [3��Ar�L�   Ar�'P   ArɘP   [3�Arɽ�   ArɘP   Ar�	P   [3�=Ar�.�   Ar�	P   Ar�zP   [4��ArΟ�   Ar�zP   Ar��P   [3�Ar��   Ar��P   Ar�\P   [3a�ArӁ�   Ar�\P   Ar��P   [4�Ar���   Ar��P   Ar�>P   [4��Ar�c�   Ar�>P   ArگP   [4Ar���   ArگP   Ar� P   [3}�Ar�E�   Ar� P   ArߑP   [3?�Ar߶�   ArߑP   Ar�P   [3�Ar�'�   Ar�P   Ar�sP   [3�Ar��   Ar�sP   Ar��P   [3��Ar�	�   Ar��P   Ar�UP   [4GRAr�z�   Ar�UP   Ar��P   [5o�Ar���   Ar��P   Ar�7P   [4��Ar�\�   Ar�7P   Ar�P   [3R�Ar���   Ar�P   Ar�P   [3Z]Ar�>�   Ar�P   Ar��P   [3�gAr���   Ar��P   Ar��P   [3Y�Ar� �   Ar��P   Ar�lP   [2�RAr���   Ar�lP   Ar��P   [2�%Ar��   Ar��P   Ar�NP   [5��Ar�s�   Ar�NP   As�P   [9RAs��   As�P   As0P   [5��AsU�   As0P   As�P   [3��As��   As�P   As	P   [3"HAs	7�   As	P   As�P   [2�As��   As�P   As�P   [3'�As�   As�P   AseP   [2��As��   AseP   As�P   [2�.As��   As�P   AsGP   [2nAsl�   AsGP   As�P   [2�	As��   As�P   As)P   [2�AsN�   As)P   As�P   [1��As��   As�P   AsP   [2O�As0�   AsP   As!|P   [2G�As!��   As!|P   As#�P   [2ivAs$�   As#�P   As&^P   [2lmAs&��   As&^P   As(�P   [1��As(��   As(�P   As+@P   [206As+e�   As+@P   As-�P   [13As-��   As-�P   As0"P   [1��As0G�   As0"P   As2�P   [1@WAs2��   As2�P   As5P   [1As5)�   As5P   As7uP   [0��As7��   As7uP   As9�P   [0F�As:�   As9�P   As<WP   [0L�As<|�   As<WP   As>�P   [04pAs>��   As>�P   AsA9P   [0�AsA^�   AsA9P   AsC�P   [0
AsC��   AsC�P   AsFP   [0
%AsF@�   AsFP   AsH�P   [.��AsH��   AsH�P   AsJ�P   [.��AsK"�   AsJ�P   AsMnP   [.�UAsM��   AsMnP   AsO�P   [.�AsP�   AsO�P   AsRPP   [/(RAsRu�   AsRPP   AsT�P   [.�AsT��   AsT�P   AsW2P   [.CAsWW�   AsW2P   AsY�P   [..�AsY��   AsY�P   As\P   [-��As\9�   As\P   As^�P   [. �As^��   As^�P   As`�P   [-�	Asa�   As`�P   AscgP   [-�Asc��   AscgP   Ase�P   [-2�Ase��   Ase�P   AshIP   [.%Ashn�   AshIP   Asj�P   [-wAsj��   Asj�P   Asm+P   [-��AsmP�   Asm+P   Aso�P   [-M�Aso��   Aso�P   AsrP   [-F�Asr2�   AsrP   Ast~P   [,��Ast��   Ast~P   Asv�P   [,>\Asw�   Asv�P   Asy`P   [,βAsy��   Asy`P   As{�P   [,>�As{��   As{�P   As~BP   [+�oAs~g�   As~BP   As��P   [,�As���   As��P   As�$P   [+�As�I�   As�$P   As��P   [+��As���   As��P   As�P   [+s�As�+�   As�P   As�wP   [,$5As���   As�wP   As��P   [+?RAs��   As��P   As�YP   [+�-As�~�   As�YP   As��P   [+(kAs���   As��P   As�;P   [*��As�`�   As�;P   As��P   [*��As���   As��P   As�P   [+$PAs�B�   As�P   As��P   [+zAs���   As��P   As��P   [+M�As�$�   As��P   As�pP   [*��As���   As�pP   As��P   [*�uAs��   As��P   As�RP   [*�As�w�   As�RP   As��P   [*�As���   As��P   As�4P   [*��As�Y�   As�4P   As��P   [*�kAs���   As��P   As�P   [*9As�;�   As�P   As��P   [*��As���   As��P   As��P   [*�fAs��   As��P   As�iP   [*��As���   As�iP   As��P   [*�As���   As��P   As�KP   [*eAs�p�   As�KP   As��P   [*q�As���   As��P   As�-P   [*��As�R�   As�-P   AsP   [*�_As���   AsP   As�P   [*�(As�4�   As�P   AsǀP   [*E$Asǥ�   AsǀP   As��P   [*��As��   As��P   As�bP   [*ěAṡ�   As�bP   As��P   [*�sAs���   As��P   As�DP   [*w�As�i�   As�DP   AsӵP   [*8�As���   AsӵP   As�&P   [)��As�K�   As�&P   AsؗP   [*JYAsؼ�   AsؗP   As�P   [*�LAs�-�   As�P   As�yP   [*�tAsݞ�   As�yP   As��P   [*�As��   As��P   As�[P   [*HAs��   As�[P   As��P   [*�As���   As��P   As�=P   [)�FAs�b�   As�=P   As�P   [)�9As���   As�P   As�P   [*#7As�D�   As�P   As�P   [)�As��   As�P   As�P   [)f�As�&�   As�P   As�rP   [)�:As��   As�rP   As��P   [)��As��   As��P   As�TP   [)*As�y�   As�TP   As��P   [)�/As���   As��P   As�6P   [)~�As�[�   As�6P   As��P   [)��As���   As��P   AtP   [)�fAt=�   AtP   At�P   [)�EAt��   At�P   At�P   [)��At�   At�P   At	kP   [)�At	��   At	kP   At�P   [)�(At�   At�P   AtMP   [)K�Atr�   AtMP   At�P   [)J�At��   At�P   At/P   [)��AtT�   At/P   At�P   [)��At��   At�P   AtP   [)��At6�   AtP   At�P   [)��At��   At�P   At�P   [)6jAt�   At�P   AtdP   [(��At��   AtdP   At!�P   [)��At!��   At!�P   At$FP   [*At$k�   At$FP   At&�P   [)+�At&��   At&�P   At)(P   [)�@At)M�   At)(P   At+�P   [)�tAt+��   At+�P   At.
P   [)��At./�   At.
P   At0{P   [)�At0��   At0{P   At2�P   [*�At3�   At2�P   At5]P   [)��At5��   At5]P   At7�P   [)�vAt7��   At7�P   At:?P   [)O0At:d�   At:?P   At<�P   [)�\At<��   At<�P   At?!P   [*?�At?F�   At?!P   AtA�P   [)��AtA��   AtA�P   AtDP   [)��AtD(�   AtDP   AtFtP   [(�AtF��   AtFtP   AtH�P   [)�AtI
�   AtH�P   AtKVP   [)MJAtK{�   AtKVP   AtM�P   [)5�AtM��   AtM�P   AtP8P   [)��AtP]�   AtP8P   AtR�P   [)R/AtR��   AtR�P   AtUP   [)��AtU?�   AtUP   AtW�P   [)hAtW��   AtW�P   AtY�P   [)(!AtZ!�   AtY�P   At\mP   [)1�At\��   At\mP   At^�P   [(ٓAt_�   At^�P   AtaOP   [))7Atat�   AtaOP   Atc�P   [*2NAtc��   Atc�P   Atf1P   [)�AtfV�   Atf1P   Ath�P   [)Y�Ath��   Ath�P   AtkP   [)�eAtk8�   AtkP   Atm�P   [)��Atm��   Atm�P   Ato�P   [)ЦAtp�   Ato�P   AtrfP   [)�<Atr��   AtrfP   Att�P   [)�Att��   Att�P   AtwHP   [)܀Atwm�   AtwHP   Aty�P   [*.�Aty��   Aty�P   At|*P   [)�~At|O�   At|*P   At~�P   [)�CAt~��   At~�P   At�P   [)�bAt�1�   At�P   At�}P   [)�At���   At�}P   At��P   [*	BAt��   At��P   At�_P   [)��At���   At�_P   At��P   [)~ZAt���   At��P   At�AP   [)�oAt�f�   At�AP   At��P   [)p�At���   At��P   At�#P   [)ͦAt�H�   At�#P   At��P   [*m%At���   At��P   At�P   [*�;At�*�   At�P   At�vP   [*�At���   At�vP   At��P   [*NAt��   At��P   At�XP   [*e[At�}�   At�XP   At��P   [*XAt���   At��P   At�:P   [*.GAt�_�   At�:P   At��P   [*��At���   At��P   At�P   [*�]At�A�   At�P   At��P   [)�At���   At��P   At��P   [*{�At�#�   At��P   At�oP   [*�At���   At�oP   At��P   [)�At��   At��P   At�QP   [*!TAt�v�   At�QP   At��P   [)z At���   At��P   At�3P   [)ʸAt�X�   At�3P   At��P   [*�At���   At��P   At�P   [*�At�:�   At�P   At��P   [*LpAt���   At��P   At��P   [*
At��   At��P   At�hP   [*S	Atō�   At�hP   At��P   [* ~At���   At��P   At�JP   [*D�At�o�   At�JP   At̻P   [*<�At���   At̻P   At�,P   [)��At�Q�   At�,P   AtѝP   [*�WAt���   AtѝP   At�P   [*<�At�3�   At�P   At�P   [*V�At֤�   At�P   At��P   [*�HAt��   At��P   At�aP   [*<RAtۆ�   At�aP   At��P   [)�\At���   At��P   At�CP   [)�nAt�h�   At�CP   At�P   [*xkAt���   At�P   At�%P   [*17At�J�   At�%P   At�P   [*�At��   At�P   At�P   [)[�At�,�   At�P   At�xP   [)��At��   At�xP   At��P   [)�{At��   At��P   At�ZP   [*�At��   At�ZP   At��P   [)��At���   At��P   At�<P   [)��At�a�   At�<P   At��P   [*uqAt���   At��P   At�P   [)�?At�C�   At�P   At��P   [*k�At���   At��P   Au  P   [)��Au %�   Au  P   AuqP   [)��Au��   AuqP   Au�P   [)��Au�   Au�P   AuSP   [*a�Aux�   AuSP   Au	�P   [*6Au	��   Au	�P   Au5P   [*^�AuZ�   Au5P   Au�P   [*��Au��   Au�P   AuP   [)�0Au<�   AuP   Au�P   [)�]Au��   Au�P   Au�P   [)�Au�   Au�P   AujP   [)��Au��   AujP   Au�P   [)ͩAu �   Au�P   AuLP   [)�Auq�   AuLP   Au�P   [)�IAu��   Au�P   Au".P   [*�Au"S�   Au".P   Au$�P   [*1�Au$��   Au$�P   Au'P   [**�Au'5�   Au'P   Au)�P   [*nDAu)��   Au)�P   Au+�P   [*��Au,�   Au+�P   Au.cP   [)�Au.��   Au.cP   Au0�P   [*:yAu0��   Au0�P   Au3EP   [*\�Au3j�   Au3EP   Au5�P   [* �Au5��   Au5�P   Au8'P   [*)�Au8L�   Au8'P   Au:�P   [*B�Au:��   Au:�P   Au=	P   [*�wAu=.�   Au=	P   Au?zP   [)�"Au?��   Au?zP   AuA�P   [)��AuB�   AuA�P   AuD\P   [)�jAuD��   AuD\P   AuF�P   [)�vAuF��   AuF�P   AuI>P   [)�"AuIc�   AuI>P   AuK�P   [*�AuK��   AuK�P   AuN P   [*0}AuNE�   AuN P   AuP�P   [*�AuP��   AuP�P   AuSP   [)�UAuS'�   AuSP   AuUsP   [)�{AuU��   AuUsP   AuW�P   [)��AuX	�   AuW�P   AuZUP   [*V�AuZz�   AuZUP   Au\�P   [)��Au\��   Au\�P   Au_7P   [)N	Au_\�   Au_7P   Aua�P   [)�RAua��   Aua�P   AudP   [*3Aud>�   AudP   Auf�P   [)�nAuf��   Auf�P   Auh�P   [*4�Aui �   Auh�P   AuklP   [*`!Auk��   AuklP   Aum�P   [*�aAun�   Aum�P   AupNP   [*:�Aups�   AupNP   Aur�P   [)��Aur��   Aur�P   Auu0P   [*~�AuuU�   Auu0P   Auw�P   [)c Auw��   Auw�P   AuzP   [(��Auz7�   AuzP   Au|�P   [)��Au|��   Au|�P   Au~�P   [)��Au�   Au~�P   Au�eP   [)��Au���   Au�eP   Au��P   [)��Au���   Au��P   Au�GP   [*dAu�l�   Au�GP   Au��P   [)��Au���   Au��P   Au�)P   [)�uAu�N�   Au�)P   Au��P   [)}8Au���   Au��P   Au�P   [)�cAu�0�   Au�P   Au�|P   [)ynAu���   Au�|P   Au��P   [(��Au��   Au��P   Au�^P   [)�nAu���   Au�^P   Au��P   [)̚Au���   Au��P   Au�@P   [)m�Au�e�   Au�@P   Au��P   [*+$Au���   Au��P   Au�"P   [*Au�G�   Au�"P   Au��P   [)�^Au���   Au��P   Au�P   [)�"Au�)�   Au�P   Au�uP   [)��Au���   Au�uP   Au��P   [)�MAu��   Au��P   Au�WP   [)��Au�|�   Au�WP   Au��P   [)+tAu���   Au��P   Au�9P   [)Au�^�   Au�9P   Au��P   [)�rAu���   Au��P   Au�P   [)C�Au�@�   Au�P   Au��P   [)`�Au���   Au��P   Au��P   [* �Au�"�   Au��P   Au�nP   [)R�Au���   Au�nP   Au��P   [)��Au��   Au��P   Au�PP   [)5Au�u�   Au�PP   Au��P   [)�AAu���   Au��P   Au�2P   [)�1Au�W�   Au�2P   AuʣP   [)�DAu���   AuʣP   Au�P   [)4�Au�9�   Au�P   AuυP   [)��AuϪ�   AuυP   Au��P   [)�Au��   Au��P   Au�gP   [)f�AuԌ�   Au�gP   Au��P   [)cMAu���   Au��P   Au�IP   [)��Au�n�   Au�IP   AuۺP   [)�1Au���   AuۺP   Au�+P   [)�Au�P�   Au�+P   Au��P   [(��Au���   Au��P   Au�P   [)��Au�2�   Au�P   Au�~P   [)l\Au��   Au�~P   Au��P   [)�Au��   Au��P   Au�`P   [)�TAu��   Au�`P   Au��P   [)B*Au���   Au��P   Au�BP   [)��Au�g�   Au�BP   Au�P   [)}g