CDF   �   
      time       bnds      lon       lat          0   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �UKESM1.0-LL (2018): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: UKCA-StratTrop
land: JULES-ES-1.0
landIce: none
ocean: NEMO-HadGEM3-GO6.0 (eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude; 75 levels; top grid cell 0-1 m)
ocnBgchem: MEDUSA2
seaIce: CICE-HadGEM3-GSI8 (eORCA1 tripolar primarily 1 deg; 360 x 330 longitude/latitude)   institution       BMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK     activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A�        creation_date         2019-04-05T16:01:49Z   
cv_version        6.2.20.1   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.MOHC.UKESM1-0-LL.historical.none.r1i1p1f2     grid      -Native N96 grid; 192 x 144 longitude/latitude      
grid_label        gn     history      =Wed Aug 10 15:18:25 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/UKESM1-0-LL_r1i1p1f2/rsdt_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp370.r1i1p1f2.Amon.rsdt.gn.v20190510/rsdt_Amon_UKESM1-0-LL_ssp370_r1i1p1f2_gn_201501-204912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp370.r1i1p1f2.Amon.rsdt.gn.v20190510/rsdt_Amon_UKESM1-0-LL_ssp370_r1i1p1f2_gn_205001-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/UKESM1-0-LL_r1i1p1f2/rsdt_UKESM1-0-LL_r1i1p1f2_ssp370.mergetime.nc
Wed Aug 10 15:18:24 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsdt/UKESM1-0-LL_r1i1p1f2/rsdt_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc
Fri Apr 08 08:52:39 2022: cdo -O -s -selname,rsdt -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 08:52:35 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsdt -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsdt/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rsdt.gn.v20190406/rsdt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc
2019-04-05T15:50:03Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
2019-04-05T15:49:43Z MIP Convert v1.0.2, Python v2.7.12, Iris v1.13.0, Numpy v1.13.3, netcdftime v1.4.1.      initialization_index            institution_id        MOHC   mip_era       CMIP6      mo_runid      u-bc179    nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      UKESM1-0-LL    parent_time_units         days since 1850-01-01-00-00-00     parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         atmos      	source_id         UKESM1-0-LL    source_type       AOGCM AER BGC CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(13 December 2018) MD5:2b12b5db6db112aa8b8b0d6c1645b121      title         %UKESM1-0-LL output prepared for CMIP6      variable_id       rsdt   variant_label         r1i1p1f2   license      XCMIP6 model data produced by the Met Office Hadley Centre is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://ukesm.ac.uk/cmip6. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   cmor_version      3.4.0      tracking_id       1hdl:21.14100/cc9cff0b-2c6b-4fc4-8674-8c9ab1bf81f6      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                   
   standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      original_name         $mo: (stash: m01s01i207, lbproc: 128)   cell_measures         area: areacella    history       u2019-04-05T16:01:49Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   C�-�Aq�6�   Aq�P   Aq��P   C�,�Aq���   Aq��P   Aq��P   C�,gAq��   Aq��P   Aq�dP   C�)�Aq���   Aq�dP   Aq��P   C�&gAq���   Aq��P   Aq�FP   C�$AAq�k�   Aq�FP   Aq��P   C�#�Aq���   Aq��P   Aq�(P   C�%/Aq�M�   Aq�(P   Aq��P   C�)oAq���   Aq��P   Aq�
P   C�,�Aq�/�   Aq�
P   Aq�{P   C�/�Aq���   Aq�{P   Aq��P   C�/Aq��   Aq��P   Aq�]P   C�,	AqĂ�   Aq�]P   Aq��P   C�*6Aq���   Aq��P   Aq�?P   C�(yAq�d�   Aq�?P   Aq˰P   C�'Aq���   Aq˰P   Aq�!P   C�&IAq�F�   Aq�!P   AqВP   C�%2Aqз�   AqВP   Aq�P   C�'�Aq�(�   Aq�P   Aq�tP   C�,Aqՙ�   Aq�tP   Aq��P   C�,rAq�
�   Aq��P   Aq�VP   C�/Aq�{�   Aq�VP   Aq��P   C�-�Aq���   Aq��P   Aq�8P   C�+SAq�]�   Aq�8P   Aq�P   C�)7Aq���   Aq�P   Aq�P   C�%KAq�?�   Aq�P   Aq�P   C�#�Aq��   Aq�P   Aq��P   C�"_Aq�!�   Aq��P   Aq�mP   C� �Aq��   Aq�mP   Aq��P   C�!:Aq��   Aq��P   Aq�OP   C�$�Aq�t�   Aq�OP   Aq��P   C�)�Aq���   Aq��P   Aq�1P   C�(�Aq�V�   Aq�1P   Aq��P   C�'�Aq���   Aq��P   Aq�P   C�-�Aq�8�   Aq�P   Aq��P   C�)�Aq���   Aq��P   Aq��P   C�%�Aq��   Aq��P   ArfP   C�#UAr��   ArfP   Ar�P   C�"UAr��   Ar�P   ArHP   C�!�Arm�   ArHP   Ar�P   C�"�Ar��   Ar�P   Ar*P   C�)3ArO�   Ar*P   Ar�P   C�.pAr��   Ar�P   ArP   C�2�Ar1�   ArP   Ar}P   C�7�Ar��   Ar}P   Ar�P   C�2�Ar�   Ar�P   Ar_P   C�,lAr��   Ar_P   Ar�P   C�(CAr��   Ar�P   ArAP   C�&dArf�   ArAP   Ar�P   C�$Ar��   Ar�P   Ar!#P   C�"�Ar!H�   Ar!#P   Ar#�P   C� zAr#��   Ar#�P   Ar&P   C�!Ar&*�   Ar&P   Ar(vP   C�%�Ar(��   Ar(vP   Ar*�P   C�-�Ar+�   Ar*�P   Ar-XP   C�*1Ar-}�   Ar-XP   Ar/�P   C�0OAr/��   Ar/�P   Ar2:P   C�,�Ar2_�   Ar2:P   Ar4�P   C�/Ar4��   Ar4�P   Ar7P   C�*�Ar7A�   Ar7P   Ar9�P   C�'�Ar9��   Ar9�P   Ar;�P   C�#9Ar<#�   Ar;�P   Ar>oP   C�!|Ar>��   Ar>oP   Ar@�P   C�!�ArA�   Ar@�P   ArCQP   C�%QArCv�   ArCQP   ArE�P   C�/EArE��   ArE�P   ArH3P   C�7LArHX�   ArH3P   ArJ�P   C�<�ArJ��   ArJ�P   ArMP   C�=�ArM:�   ArMP   ArO�P   C�5MArO��   ArO�P   ArQ�P   C�-�ArR�   ArQ�P   ArThP   C�(�ArT��   ArThP   ArV�P   C�$dArV��   ArV�P   ArYJP   C�#hArYo�   ArYJP   Ar[�P   C�%NAr[��   Ar[�P   Ar^,P   C�*�Ar^Q�   Ar^,P   Ar`�P   C�1�Ar`��   Ar`�P   ArcP   C�7�Arc3�   ArcP   AreP   C�3�Are��   AreP   Arg�P   C�1�Arh�   Arg�P   ArjaP   C�1�Arj��   ArjaP   Arl�P   C�,2Arl��   Arl�P   AroCP   C�(!Aroh�   AroCP   Arq�P   C�&XArq��   Arq�P   Art%P   C�(SArtJ�   Art%P   Arv�P   C�.�Arv��   Arv�P   AryP   C�=^Ary,�   AryP   Ar{xP   C�>cAr{��   Ar{xP   Ar}�P   C�;�Ar~�   Ar}�P   Ar�ZP   C�;Ar��   Ar�ZP   Ar��P   C�6Ar���   Ar��P   Ar�<P   C�2*Ar�a�   Ar�<P   Ar��P   C�-hAr���   Ar��P   Ar�P   C�(�Ar�C�   Ar�P   Ar��P   C�* Ar���   Ar��P   Ar� P   C�2�Ar�%�   Ar� P   Ar�qP   C�7Ar���   Ar�qP   Ar��P   C�B�Ar��   Ar��P   Ar�SP   C�F�Ar�x�   Ar�SP   Ar��P   C�B�Ar���   Ar��P   Ar�5P   C�:HAr�Z�   Ar�5P   Ar��P   C�/IAr���   Ar��P   Ar�P   C�.�Ar�<�   Ar�P   Ar��P   C�+wAr���   Ar��P   Ar��P   C�+zAr��   Ar��P   Ar�jP   C�2PAr���   Ar�jP   Ar��P   C�A;Ar� �   Ar��P   Ar�LP   C�O�Ar�q�   Ar�LP   Ar��P   C�Q7Ar���   Ar��P   Ar�.P   C�I_Ar�S�   Ar�.P   Ar��P   C�C�Ar���   Ar��P   Ar�P   C�7�Ar�5�   Ar�P   Ar��P   C�/gAr���   Ar��P   Ar��P   C�,�Ar��   Ar��P   Ar�cP   C�+>Ar���   Ar�cP   Ar��P   C�+�Ar���   Ar��P   Ar�EP   C�2SAr�j�   Ar�EP   ArĶP   C�:ZAr���   ArĶP   Ar�'P   C�>�Ar�L�   Ar�'P   ArɘP   C�??Arɽ�   ArɘP   Ar�	P   C�?�Ar�.�   Ar�	P   Ar�zP   C�8�ArΟ�   Ar�zP   Ar��P   C�8Ar��   Ar��P   Ar�\P   C�0�ArӁ�   Ar�\P   Ar��P   C�-dAr���   Ar��P   Ar�>P   C�-wAr�c�   Ar�>P   ArگP   C�,KAr���   ArگP   Ar� P   C�3�Ar�E�   Ar� P   ArߑP   C�?9Ar߶�   ArߑP   Ar�P   C�IAr�'�   Ar�P   Ar�sP   C�GAr��   Ar�sP   Ar��P   C�G�Ar�	�   Ar��P   Ar�UP   C�<�Ar�z�   Ar�UP   Ar��P   C�<�Ar���   Ar��P   Ar�7P   C�/�Ar�\�   Ar�7P   Ar�P   C�+oAr���   Ar�P   Ar�P   C�+�Ar�>�   Ar�P   Ar��P   C�.�Ar���   Ar��P   Ar��P   C�: Ar� �   Ar��P   Ar�lP   C�IdAr���   Ar�lP   Ar��P   C�F�Ar��   Ar��P   Ar�NP   C�FAr�s�   Ar�NP   As�P   C�A
As��   As�P   As0P   C�4�AsU�   As0P   As�P   C�-`As��   As�P   As	P   C�+�As	7�   As	P   As�P   C�(�As��   As�P   As�P   C�,�As�   As�P   AseP   C�8�As��   AseP   As�P   C�AMAs��   As�P   AsGP   C�G�Asl�   AsGP   As�P   C�EAs��   As�P   As)P   C�F�AsN�   As)P   As�P   C�7As��   As�P   AsP   C�0TAs0�   AsP   As!|P   C�)�As!��   As!|P   As#�P   C�(As$�   As#�P   As&^P   C�%8As&��   As&^P   As(�P   C�$�As(��   As(�P   As+@P   C�$�As+e�   As+@P   As-�P   C�*&As-��   As-�P   As0"P   C�2�As0G�   As0"P   As2�P   C�6eAs2��   As2�P   As5P   C�8YAs5)�   As5P   As7uP   C�:�As7��   As7uP   As9�P   C�3oAs:�   As9�P   As<WP   C�*(As<|�   As<WP   As>�P   C�'�As>��   As>�P   AsA9P   C�&�AsA^�   AsA9P   AsC�P   C�&�AsC��   AsC�P   AsFP   C�'�AsF@�   AsFP   AsH�P   C�0�AsH��   AsH�P   AsJ�P   C�:�AsK"�   AsJ�P   AsMnP   C�@�AsM��   AsMnP   AsO�P   C�B�AsP�   AsO�P   AsRPP   C�=\AsRu�   AsRPP   AsT�P   C�7AsT��   AsT�P   AsW2P   C�0�AsWW�   AsW2P   AsY�P   C�.%AsY��   AsY�P   As\P   C�)�As\9�   As\P   As^�P   C�'�As^��   As^�P   As`�P   C�%^Asa�   As`�P   AscgP   C�%�Asc��   AscgP   Ase�P   C�+OAse��   Ase�P   AshIP   C�4�Ashn�   AshIP   Asj�P   C�5�Asj��   Asj�P   Asm+P   C�:ZAsmP�   Asm+P   Aso�P   C�5�Aso��   Aso�P   AsrP   C�5�Asr2�   AsrP   Ast~P   C�/�Ast��   Ast~P   Asv�P   C�)�Asw�   Asv�P   Asy`P   C�$�Asy��   Asy`P   As{�P   C�#uAs{��   As{�P   As~BP   C�$$As~g�   As~BP   As��P   C�'@As���   As��P   As�$P   C�0LAs�I�   As�$P   As��P   C�6pAs���   As��P   As�P   C�=?As�+�   As�P   As�wP   C�>As���   As�wP   As��P   C�8WAs��   As��P   As�YP   C�/�As�~�   As�YP   As��P   C�+�As���   As��P   As�;P   C�'�As�`�   As�;P   As��P   C�$�As���   As��P   As�P   C�%�As�B�   As�P   As��P   C�+As���   As��P   As��P   C�1�As�$�   As��P   As�pP   C�1�As���   As�pP   As��P   C�2�As��   As��P   As�RP   C�3�As�w�   As�RP   As��P   C�.5As���   As��P   As�4P   C�)�As�Y�   As�4P   As��P   C�%�As���   As��P   As�P   C�$�As�;�   As�P   As��P   C�$As���   As��P   As��P   C�%�As��   As��P   As�iP   C�,�As���   As�iP   As��P   C�3*As���   As��P   As�KP   C�6�As�p�   As�KP   As��P   C�5UAs���   As��P   As�-P   C�1sAs�R�   As�-P   AsP   C�+�As���   AsP   As�P   C�)As�4�   As�P   AsǀP   C�'Asǥ�   AsǀP   As��P   C�$As��   As��P   As�bP   C�"�Aṡ�   As�bP   As��P   C�!�As���   As��P   As�DP   C�#=As�i�   As�DP   AsӵP   C�'�As���   AsӵP   As�&P   C�,�As�K�   As�&P   AsؗP   C�0Asؼ�   AsؗP   As�P   C�0OAs�-�   As�P   As�yP   C�07Asݞ�   As�yP   As��P   C�,�As��   As��P   As�[P   C�*�As��   As�[P   As��P   C�%qAs���   As��P   As�=P   C�"sAs�b�   As�=P   As�P   C�"2As���   As�P   As�P   C�"�As�D�   As�P   As�P   C�&�As��   As�P   As�P   C�-�As�&�   As�P   As�rP   C�2�As��   As�rP   As��P   C�9xAs��   As��P   As�TP   C�6�As�y�   As�TP   As��P   C�1&As���   As��P   As�6P   C�,�As�[�   As�6P   As��P   C�'�As���   As��P   AtP   C�%�At=�   AtP   At�P   C�$(At��   At�P   At�P   C�&�At�   At�P   At	kP   C�,