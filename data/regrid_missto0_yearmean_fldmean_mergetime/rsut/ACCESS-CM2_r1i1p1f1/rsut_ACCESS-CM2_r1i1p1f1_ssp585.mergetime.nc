CDF  �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
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
grid_label        gn     history      
�Wed Aug 10 15:18:28 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp585.r1i1p1f1.Amon.rsut.gn.v20210317/rsut_Amon_ACCESS-CM2_ssp585_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp585.r1i1p1f1.Amon.rsut.gn.v20210317/rsut_Amon_ACCESS-CM2_ssp585_r1i1p1f1_gn_210101-230012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_ssp585.mergetime.nc
Wed Aug 10 15:18:26 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/ACCESS-CM2_r1i1p1f1/rsut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 08:53:13 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 08:53:09 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rsut.gn.v20191108/rsut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-11-08T10:09:32Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: CM2-historical; Local ID: bj594; Variable: rsut (['fld_s01i208'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0950-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   version       	v20191108      cmor_version      3.4.0      tracking_id       1hdl:21.14100/ae0d2880-10cc-4234-90ec-f56309c77f6a      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   cell_measures         area: areacella    history       u2019-11-08T10:09:31Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   B�2�Aq�6�   Aq�P   Aq��P   B�Aq���   Aq��P   Aq��P   B�'AAq��   Aq��P   Aq�dP   B��!Aq���   Aq�dP   Aq��P   BÄ�Aq���   Aq��P   Aq�FP   B�
Aq�k�   Aq�FP   Aq��P   B�8�Aq���   Aq��P   Aq�(P   BÊ�Aq�M�   Aq�(P   Aq��P   BõdAq���   Aq��P   Aq�
P   B��Aq�/�   Aq�
P   Aq�{P   BÖvAq���   Aq�{P   Aq��P   B�SAq��   Aq��P   Aq�]P   B�Z�AqĂ�   Aq�]P   Aq��P   B���Aq���   Aq��P   Aq�?P   B�,tAq�d�   Aq�?P   Aq˰P   B�n�Aq���   Aq˰P   Aq�!P   B�5Aq�F�   Aq�!P   AqВP   Bā�Aqз�   AqВP   Aq�P   B�ýAq�(�   Aq�P   Aq�tP   B�Aqՙ�   Aq�tP   Aq��P   B��%Aq�
�   Aq��P   Aq�VP   B�9�Aq�{�   Aq�VP   Aq��P   B��Aq���   Aq��P   Aq�8P   B�kAq�]�   Aq�8P   Aq�P   B�"�Aq���   Aq�P   Aq�P   B�O8Aq�?�   Aq�P   Aq�P   BË�Aq��   Aq�P   Aq��P   B�F�Aq�!�   Aq��P   Aq�mP   B��LAq��   Aq�mP   Aq��P   B�´Aq��   Aq��P   Aq�OP   B��Aq�t�   Aq�OP   Aq��P   B�\Aq���   Aq��P   Aq�1P   B�pAq�V�   Aq�1P   Aq��P   Bǭ<Aq���   Aq��P   Aq�P   Bˀ�Aq�8�   Aq�P   Aq��P   B�"pAq���   Aq��P   Aq��P   B��%Aq��   Aq��P   ArfP   B��7Ar��   ArfP   Ar�P   B���Ar��   Ar�P   ArHP   B�6PArm�   ArHP   Ar�P   B�>�Ar��   Ar�P   Ar*P   B�6ArO�   Ar*P   Ar�P   BŀuAr��   Ar�P   ArP   Bķ�Ar1�   ArP   Ar}P   B���Ar��   Ar}P   Ar�P   B��XAr�   Ar�P   Ar_P   B��~Ar��   Ar_P   Ar�P   B�pAr��   Ar�P   ArAP   B�J�Arf�   ArAP   Ar�P   B�׹Ar��   Ar�P   Ar!#P   Bö�Ar!H�   Ar!#P   Ar#�P   B�Ar#��   Ar#�P   Ar&P   B���Ar&*�   Ar&P   Ar(vP   BǛ�Ar(��   Ar(vP   Ar*�P   B�M6Ar+�   Ar*�P   Ar-XP   B�ղAr-}�   Ar-XP   Ar/�P   B���Ar/��   Ar/�P   Ar2:P   B�t�Ar2_�   Ar2:P   Ar4�P   B�7
Ar4��   Ar4�P   Ar7P   B�+lAr7A�   Ar7P   Ar9�P   BĔ�Ar9��   Ar9�P   Ar;�P   B�ΊAr<#�   Ar;�P   Ar>oP   B�ApAr>��   Ar>oP   Ar@�P   BƸ�ArA�   Ar@�P   ArCQP   BƇ,ArCv�   ArCQP   ArE�P   B�`�ArE��   ArE�P   ArH3P   B�u�ArHX�   ArH3P   ArJ�P   BſKArJ��   ArJ�P   ArMP   B��ZArM:�   ArMP   ArO�P   B�ntArO��   ArO�P   ArQ�P   B�\�ArR�   ArQ�P   ArThP   B�c�ArT��   ArThP   ArV�P   B�SQArV��   ArV�P   ArYJP   B�:RArYo�   ArYJP   Ar[�P   B��3Ar[��   Ar[�P   Ar^,P   B��KAr^Q�   Ar^,P   Ar`�P   B�=�Ar`��   Ar`�P   ArcP   B�[�Arc3�   ArcP   AreP   B�G.Are��   AreP   Arg�P   B�FArh�   Arg�P   ArjaP   B�Arj��   ArjaP   Arl�P   B�ݽArl��   Arl�P   AroCP   B�
�Aroh�   AroCP   Arq�P   BýaArq��   Arq�P   Art%P   B�ڻArtJ�   Art%P   Arv�P   B�%�Arv��   Arv�P   AryP   BďBAry,�   AryP   Ar{xP   BįoAr{��   Ar{xP   Ar}�P   B�aAr~�   Ar}�P   Ar�ZP   B��?Ar��   Ar�ZP   Ar��P   B�CAr���   Ar��P   Ar�<P   B�a*Ar�a�   Ar�<P   Ar��P   B�)Ar���   Ar��P   Ar�P   BīZAr�C�   Ar�P   Ar��P   B��Ar���   Ar��P   Ar� P   BÊ�Ar�%�   Ar� P   Ar�qP   B�M�Ar���   Ar�qP   Ar��P   B��Ar��   Ar��P   Ar�SP   B�2}Ar�x�   Ar�SP   Ar��P   B�=JAr���   Ar��P   Ar�5P   B�9#Ar�Z�   Ar�5P   Ar��P   Bĳ3Ar���   Ar��P   Ar�P   Bĺ�Ar�<�   Ar�P   Ar��P   B�n)Ar���   Ar��P   Ar��P   B�y�Ar��   Ar��P   Ar�jP   B�1�Ar���   Ar�jP   Ar��P   B�t@Ar� �   Ar��P   Ar�LP   B�"�Ar�q�   Ar�LP   Ar��P   B�E�Ar���   Ar��P   Ar�.P   B��}Ar�S�   Ar�.P   Ar��P   B�88Ar���   Ar��P   Ar�P   B��{Ar�5�   Ar�P   Ar��P   B�Y=Ar���   Ar��P   Ar��P   B���Ar��   Ar��P   Ar�cP   BȉQAr���   Ar�cP   Ar��P   B���Ar���   Ar��P   Ar�EP   BƁAr�j�   Ar�EP   ArĶP   B��.Ar���   ArĶP   Ar�'P   B�RkAr�L�   Ar�'P   ArɘP   BƁArɽ�   ArɘP   Ar�	P   BƙZAr�.�   Ar�	P   Ar�zP   B�f3ArΟ�   Ar�zP   Ar��P   B� �Ar��   Ar��P   Ar�\P   B���ArӁ�   Ar�\P   Ar��P   BƩ�Ar���   Ar��P   Ar�>P   B�a�Ar�c�   Ar�>P   ArگP   Bƹ3Ar���   ArگP   Ar� P   B��Ar�E�   Ar� P   ArߑP   B��IAr߶�   ArߑP   Ar�P   B�G�Ar�'�   Ar�P   Ar�sP   B�N7Ar��   Ar�sP   Ar��P   B�_�Ar�	�   Ar��P   Ar�UP   B��Ar�z�   Ar�UP   Ar��P   B�<>Ar���   Ar��P   Ar�7P   B�=7Ar�\�   Ar�7P   Ar�P   B��VAr���   Ar�P   Ar�P   B��Ar�>�   Ar�P   Ar��P   Bƕ<Ar���   Ar��P   Ar��P   B��Ar� �   Ar��P   Ar�lP   B�l[Ar���   Ar�lP   Ar��P   B�VAr��   Ar��P   Ar�NP   B��Ar�s�   Ar�NP   As�P   B̅�As��   As�P   As0P   B�ǀAsU�   As0P   As�P   BƟ�As��   As�P   As	P   Bű�As	7�   As	P   As�P   B��As��   As�P   As�P   BŷzAs�   As�P   AseP   B��As��   AseP   As�P   B�EAs��   As�P   AsGP   B�tAsl�   AsGP   As�P   B�As��   As�P   As)P   BĆ:AsN�   As)P   As�P   B�`�As��   As�P   AsP   B�ɇAs0�   AsP   As!|P   B��:As!��   As!|P   As#�P   B���As$�   As#�P   As&^P   B�� As&��   As&^P   As(�P   B�7�As(��   As(�P   As+@P   BĦ�As+e�   As+@P   As-�P   BÏLAs-��   As-�P   As0"P   B��2As0G�   As0"P   As2�P   BÝ�As2��   As2�P   As5P   B��As5)�   As5P   As7uP   B���As7��   As7uP   As9�P   B��As:�   As9�P   As<WP   B���As<|�   As<WP   As>�P   B�WAs>��   As>�P   AsA9P   B��4AsA^�   AsA9P   AsC�P   B��AsC��   AsC�P   AsFP   B�1�AsF@�   AsFP   AsH�P   B��AsH��   AsH�P   AsJ�P   B�C4AsK"�   AsJ�P   AsMnP   B���AsM��   AsMnP   AsO�P   B�5}AsP�   AsO�P   AsRPP   B��_AsRu�   AsRPP   AsT�P   B�w�AsT��   AsT�P   AsW2P   B��_AsWW�   AsW2P   AsY�P   B��FAsY��   AsY�P   As\P   B��sAs\9�   As\P   As^�P   B��ZAs^��   As^�P   As`�P   B���Asa�   As`�P   AscgP   B�])Asc��   AscgP   Ase�P   B��fAse��   Ase�P   AshIP   B��uAshn�   AshIP   Asj�P   B��Asj��   Asj�P   Asm+P   B��NAsmP�   Asm+P   Aso�P   B�K!Aso��   Aso�P   AsrP   B��Asr2�   AsrP   Ast~P   B�-Ast��   Ast~P   Asv�P   B�< Asw�   Asv�P   Asy`P   B���Asy��   Asy`P   As{�P   B���As{��   As{�P   As~BP   B��5As~g�   As~BP   As��P   B�2:As���   As��P   As�$P   B�wAs�I�   As�$P   As��P   B��'As���   As��P   As�P   B��As�+�   As�P   As�wP   B�OAs���   As�wP   As��P   B��
As��   As��P   As�YP   B��lAs�~�   As�YP   As��P   B���As���   As��P   As�;P   B��ZAs�`�   As�;P   As��P   B�kAs���   As��P   As�P   B�ՉAs�B�   As�P   As��P   B�OMAs���   As��P   As��P   B���As�$�   As��P   As�pP   B�eAs���   As�pP   As��P   B��As��   As��P   As�RP   B�	#As�w�   As�RP   As��P   B��As���   As��P   As�4P   B��5As�Y�   As�4P   As��P   B���As���   As��P   As�P   B�GAs�;�   As�P   As��P   B��As���   As��P   As��P   B�@�As��   As��P   As�iP   B�+�As���   As�iP   As��P   B�IAs���   As��P   As�KP   B�g�As�p�   As�KP   As��P   B�s�As���   As��P   As�-P   B���As�R�   As�-P   AsP   B�'�As���   AsP   As�P   B�'�As�4�   As�P   AsǀP   B�AAsǥ�   AsǀP   As��P   B��VAs��   As��P   As�bP   B�|�Aṡ�   As�bP   As��P   B���As���   As��P   As�DP   B�N~As�i�   As�DP   AsӵP   B�F�As���   AsӵP   As�&P   B���As�K�   As�&P   AsؗP   B��Asؼ�   AsؗP   As�P   B�$�As�-�   As�P   As�yP   B��Asݞ�   As�yP   As��P   B��"As��   As��P   As�[P   B�yAs��   As�[P   As��P   B���As���   As��P   As�=P   B�W�As�b�   As�=P   As�P   B�'�As���   As�P   As�P   B�r�As�D�   As�P   As�P   B�
�As��   As�P   As�P   B�OAs�&�   As�P   As�rP   B��1As��   As�rP   As��P   B�j�As��   As��P   As�TP   B��zAs�y�   As�TP   As��P   B�%�As���   As��P   As�6P   B�_�As�[�   As�6P   As��P   B��8As���   As��P   AtP   B��|At=�   AtP   At�P   B��At��   At�P   At�P   B���At�   At�P   At	kP   B�At	��   At	kP   At�P   B�I�At�   At�P   AtMP   B���Atr�   AtMP   At�P   B���At��   At�P   At/P   B�|�AtT�   At/P   At�P   B�78At��   At�P   AtP   B�P At6�   AtP   At�P   B�\9At��   At�P   At�P   B�uAt�   At�P   AtdP   B�fDAt��   AtdP   At!�P   B�.ZAt!��   At!�P   At$FP   B�}|At$k�   At$FP   At&�P   B���At&��   At&�P   At)(P   B��QAt)M�   At)(P   At+�P   B���At+��   At+�P   At.
P   B���At./�   At.
P   At0{P   B�v5At0��   At0{P   At2�P   B���At3�   At2�P   At5]P   B�ʧAt5��   At5]P   At7�P   B���At7��   At7�P   At:?P   B�g At:d�   At:?P   At<�P   B�y6At<��   At<�P   At?!P   B���At?F�   At?!P   AtA�P   B�2AtA��   AtA�P   AtDP   B�i�AtD(�   AtDP   AtFtP   B��AtF��   AtFtP   AtH�P   B���AtI
�   AtH�P   AtKVP   B�eAtK{�   AtKVP   AtM�P   B���AtM��   AtM�P   AtP8P   B��AtP]�   AtP8P   AtR�P   B���AtR��   AtR�P   AtUP   B���AtU?�   AtUP   AtW�P   B��AtW��   AtW�P   AtY�P   B�yAtZ!�   AtY�P   At\mP   B��HAt\��   At\mP   At^�P   B���At_�   At^�P   AtaOP   B���Atat�   AtaOP   Atc�P   B��EAtc��   Atc�P   Atf1P   B�rAtfV�   Atf1P   Ath�P   B��~Ath��   Ath�P   AtkP   B�m�Atk8�   AtkP   Atm�P   B���Atm��   Atm�P   Ato�P   B��QAtp�   Ato�P   AtrfP   B��Atr��   AtrfP   Att�P   B��Att��   Att�P   AtwHP   B�XAtwm�   AtwHP   Aty�P   B��~Aty��   Aty�P   At|*P   B�\CAt|O�   At|*P   At~�P   B�;�At~��   At~�P   At�P   B���At�1�   At�P   At�}P   B�HAt���   At�}P   At��P   B��At��   At��P   At�_P   B�f�At���   At�_P   At��P   B�eAt���   At��P   At�AP   B��}At�f�   At�AP   At��P   B�->At���   At��P   At�#P   B���At�H�   At�#P   At��P   B�;DAt���   At��P   At�P   B��DAt�*�   At�P   At�vP   B��At���   At�vP   At��P   B��=At��   At��P   At�XP   B��At�}�   At�XP   At��P   B��%At���   At��P   At�:P   B��At�_�   At�:P   At��P   B��TAt���   At��P   At�P   B��At�A�   At�P   At��P   B�P�At���   At��P   At��P   B�þAt�#�   At��P   At�oP   B�\�At���   At�oP   At��P   B�7�At��   At��P   At�QP   B���At�v�   At�QP   At��P   B��At���   At��P   At�3P   B��<At�X�   At�3P   At��P   B���At���   At��P   At�P   B��lAt�:�   At�P   At��P   B�,�At���   At��P   At��P   B�P|At��   At��P   At�hP   B��Atō�   At�hP   At��P   B�%At���   At��P   At�JP   B��At�o�   At�JP   At̻P   B���At���   At̻P   At�,P   B��At�Q�   At�,P   AtѝP   B�+MAt���   AtѝP   At�P   B��At�3�   At�P   At�P   B��At֤�   At�P   At��P   B��/At��   At��P   At�aP   B�;-Atۆ�   At�aP   At��P   B�a�At���   At��P   At�CP   B���At�h�   At�CP   At�P   B��YAt���   At�P   At�%P   B���At�J�   At�%P   At�P   B���At��   At�P   At�P   B���At�,�   At�P   At�xP   B�,�At��   At�xP   At��P   B�8mAt��   At��P   At�ZP   B�j�At��   At�ZP   At��P   B�4�At���   At��P   At�<P   B�[At�a�   At�<P   At��P   B��6At���   At��P   At�P   B��sAt�C�   At�P   At��P   B�GAt���   At��P   Au  P   B�	�Au %�   Au  P   AuqP   B��?Au��   AuqP   Au�P   B�GlAu�   Au�P   AuSP   B��<Aux�   AuSP   Au	�P   B���Au	��   Au	�P   Au5P   B���AuZ�   Au5P   Au�P   B��Au��   Au�P   AuP   B��Au<�   AuP   Au�P   B��_Au��   Au�P   Au�P   B� �Au�   Au�P   AujP   B�rGAu��   AujP   Au�P   B�8lAu �   Au�P   AuLP   B�^Auq�   AuLP   Au�P   B�� Au��   Au�P   Au".P   B���Au"S�   Au".P   Au$�P   B�L�Au$��   Au$�P   Au'P   B��Au'5�   Au'P   Au)�P   B�_Au)��   Au)�P   Au+�P   B�� Au,�   Au+�P   Au.cP   B�!�Au.��   Au.cP   Au0�P   B��Au0��   Au0�P   Au3EP   B�&Au3j�   Au3EP   Au5�P   B�Au5��   Au5�P   Au8'P   B��fAu8L�   Au8'P   Au:�P   B�OIAu:��   Au:�P   Au=	P   B��:Au=.�   Au=	P   Au?zP   B�?Au?��   Au?zP   AuA�P   B���AuB�   AuA�P   AuD\P   B�lAuD��   AuD\P   AuF�P   B�2{AuF��   AuF�P   AuI>P   B���AuIc�   AuI>P   AuK�P   B�=NAuK��   AuK�P   AuN P   B�EGAuNE�   AuN P   AuP�P   B�=CAuP��   AuP�P   AuSP   B��AuS'�   AuSP   AuUsP   B���AuU��   AuUsP   AuW�P   B��8AuX	�   AuW�P   AuZUP   B�CAuZz�   AuZUP   Au\�P   B�%�Au\��   Au\�P   Au_7P   B��EAu_\�   Au_7P   Aua�P   B�D�Aua��   Aua�P   AudP   B���Aud>�   AudP   Auf�P   B�b2Auf��   Auf�P   Auh�P   B��Aui �   Auh�P   AuklP   B�R�Auk��   AuklP   Aum�P   B�v�Aun�   Aum�P   AupNP   B��Aups�   AupNP   Aur�P   B��VAur��   Aur�P   Auu0P   B�5�AuuU�   Auu0P   Auw�P   B�[�Auw��   Auw�P   AuzP   B�V�Auz7�   AuzP   Au|�P   B���Au|��   Au|�P   Au~�P   B�dAu�   Au~�P   Au�eP   B��Au���   Au�eP   Au��P   B�&�Au���   Au��P   Au�GP   B�l�Au�l�   Au�GP   Au��P   B�YvAu���   Au��P   Au�)P   B�}�Au�N�   Au�)P   Au��P   B�5/Au���   Au��P   Au�P   B�)�Au�0�   Au�P   Au�|P   B�Au���   Au�|P   Au��P   B�KAu��   Au��P   Au�^P   B��Au���   Au�^P   Au��P   B���Au���   Au��P   Au�@P   B���Au�e�   Au�@P   Au��P   B��Au���   Au��P   Au�"P   B��Au�G�   Au�"P   Au��P   B��*Au���   Au��P   Au�P   B���Au�)�   Au�P   Au�uP   B��Au���   Au�uP   Au��P   B�oAu��   Au��P   Au�WP   B��.Au�|�   Au�WP   Au��P   B���Au���   Au��P   Au�9P   B�P(Au�^�   Au�9P   Au��P   B��Au���   Au��P   Au�P   B��]Au�@�   Au�P   Au��P   B��Au���   Au��P   Au��P   B���Au�"�   Au��P   Au�nP   B�qAu���   Au�nP   Au��P   B�>aAu��   Au��P   Au�PP   B��YAu�u�   Au�PP   Au��P   B��Au���   Au��P   Au�2P   B�E�Au�W�   Au�2P   AuʣP   B��/Au���   AuʣP   Au�P   B��Au�9�   Au�P   AuυP   B��yAuϪ�   AuυP   Au��P   B��MAu��   Au��P   Au�gP   B�>�AuԌ�   Au�gP   Au��P   B��sAu���   Au��P   Au�IP   B���Au�n�   Au�IP   AuۺP   B�6bAu���   AuۺP   Au�+P   B�3dAu�P�   Au�+P   Au��P   B��(Au���   Au��P   Au�P   B�(�Au�2�   Au�P   Au�~P   B�_zAu��   Au�~P   Au��P   B��Au��   Au��P   Au�`P   B�Au��   Au�`P   Au��P   B�y Au���   Au��P   Au�BP   B�=pAu�g�   Au�BP   Au�P   B�2�