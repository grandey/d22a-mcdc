CDF   �   
      time       bnds      lon       lat          -   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       0MIROC6 (2017): 
aerosol: SPRINTARS6.0
atmos: CCSR AGCM (T85; 256 x 128 longitude/latitude; 81 levels; top level 0.004 hPa)
atmosChem: none
land: MATSIRO6.0
landIce: none
ocean: COCO4.9 (tripolar primarily 1deg; 360 x 256 longitude/latitude; 63 levels; top grid cell 0-2 m)
ocnBgchem: none
seaIce: COCO4.9   institution      QJAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2020-01-29T22:53:34Z   data_specs_version        01.00.31   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     forcing_index               	frequency         year   further_info_url      Jhttps://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.historical.none.r1i1p1f1     grid      -native ocean tripolar grid with 360x256 cells      
grid_label        gm     history      dWed Aug 10 15:22:30 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/MIROC6_r1i1p1f1/zostoga_MIROC6_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/MIROC6_r1i1p1f1/CMIP6.ScenarioMIP.MIROC.MIROC6.ssp585.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_ssp585_r1i1p1f1_gm_201501-210012.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/MIROC6_r1i1p1f1/zostoga_MIROC6_r1i1p1f1_ssp585.mergetime.nc
Wed Aug 10 15:22:29 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_historical_r1i1p1f1_gm_185001-201412.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/MIROC6_r1i1p1f1/zostoga_MIROC6_r1i1p1f1_historical.mergetime.nc
Thu Apr 07 23:33:18 2022: cdo -O -s -selname,zostoga -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_historical_r1i1p1f1_gm_185001-201412.1d.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_historical_r1i1p1f1_gm_185001-201412.1d.yearmean.fldmean.nc
Thu Apr 07 23:33:17 2022: cdo -O -s --reduce_dim -selname,zostoga -yearmean /Users/benjamin/Data/p22b/CMIP6/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_historical_r1i1p1f1_gm_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_historical_r1i1p1f1_gm_185001-201412.1d.yearmean.nc
2020-01-29T22:53:34Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.   initialization_index            institution_id        MIROC      mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      MIROC6     parent_time_units         days since 3200-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      	source_id         MIROC6     source_type       	AOGCM AER      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ACreation Date:(22 July 2019) MD5:b4cefb4b6dbb146fea9677a552a00934      title          MIROC6 output prepared for CMIP6   variable_id       zostoga    variant_label         r1i1p1f1   license      !CMIP6 model data produced by MIROC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      cmor_version      3.5.0      tracking_id       1hdl:21.14100/dd45d71b-b5b5-447f-9894-78b0086f02ff      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   zostoga                    
   standard_name         ,global_average_thermosteric_sea_level_change   	long_name         ,Global Average Thermosteric Sea Level Change   units         m      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       /There is no CMIP6 request for zosga nor zossga.    original_name         shstg      original_units        cm     history       �2020-01-29T22:53:34Z altered by CMOR: Converted units from 'cm' to 'm'. 2020-01-29T22:53:34Z altered by CMOR: replaced missing value flag (-999) and corresponding data with standard missing value (1e+20).            �                Aq���   Aq��P   Aq�P   9���Aq�6�   Aq�P   Aq��P   :4�Aq���   Aq��P   Aq��P   :�-�Aq��   Aq��P   Aq�dP   ;�Aq���   Aq�dP   Aq��P   ;v%�Aq���   Aq��P   Aq�FP   :ћ�Aq�k�   Aq�FP   Aq��P   9�aAq���   Aq��P   Aq�(P   :�h�Aq�M�   Aq�(P   Aq��P   ;j�Aq���   Aq��P   Aq�
P   ;[gAq�/�   Aq�
P   Aq�{P   ;I{�Aq���   Aq�{P   Aq��P   ;S6�Aq��   Aq��P   Aq�]P   ;vAqĂ�   Aq�]P   Aq��P   �FAq���   Aq��P   Aq�?P   ����Aq�d�   Aq�?P   Aq˰P   ��d�Aq���   Aq˰P   Aq�!P   :�C�Aq�F�   Aq�!P   AqВP   ;X?�Aqз�   AqВP   Aq�P   ;��Aq�(�   Aq�P   Aq�tP   ;�{�Aqՙ�   Aq�tP   Aq��P   ;\]Aq�
�   Aq��P   Aq�VP   ;f:Aq�{�   Aq�VP   Aq��P   ;�
0Aq���   Aq��P   Aq�8P   ;���Aq�]�   Aq�8P   Aq�P   ;��9Aq���   Aq�P   Aq�P   ;}�Aq�?�   Aq�P   Aq�P   ;�w�Aq��   Aq�P   Aq��P   ;��Aq�!�   Aq��P   Aq�mP   ;�J�Aq��   Aq�mP   Aq��P   ;��Aq��   Aq��P   Aq�OP   ;��Aq�t�   Aq�OP   Aq��P   ;���Aq���   Aq��P   Aq�1P   ;�QAq�V�   Aq�1P   Aq��P   ;��Aq���   Aq��P   Aq�P   ;�~`Aq�8�   Aq�P   Aq��P   ;E�sAq���   Aq��P   Aq��P   ;f�LAq��   Aq��P   ArfP   ;�<4Ar��   ArfP   Ar�P   ;��Ar��   Ar�P   ArHP   ;��Arm�   ArHP   Ar�P   ;�KAr��   Ar�P   Ar*P   ;��ArO�   Ar*P   Ar�P   ;��UAr��   Ar�P   ArP   ;�s%Ar1�   ArP   Ar}P   ;�iAr��   Ar}P   Ar�P   ;��~Ar�   Ar�P   Ar_P   ;�Ar��   Ar_P   Ar�P   < ��Ar��   Ar�P   ArAP   ;�۠Arf�   ArAP   Ar�P   ;���Ar��   Ar�P   Ar!#P   <JAr!H�   Ar!#P   Ar#�P   ;�1�Ar#��   Ar#�P   Ar&P   ;��3Ar&*�   Ar&P   Ar(vP   ;�N Ar(��   Ar(vP   Ar*�P   ;�� Ar+�   Ar*�P   Ar-XP   ;�>�Ar-}�   Ar-XP   Ar/�P   <��Ar/��   Ar/�P   Ar2:P   <	�Ar2_�   Ar2:P   Ar4�P   <�Ar4��   Ar4�P   Ar7P   <g�Ar7A�   Ar7P   Ar9�P   <�Ar9��   Ar9�P   Ar;�P   <Z�Ar<#�   Ar;�P   Ar>oP   <�Ar>��   Ar>oP   Ar@�P   <lPArA�   Ar@�P   ArCQP   <	��ArCv�   ArCQP   ArE�P   <�5ArE��   ArE�P   ArH3P   <&� ArHX�   ArH3P   ArJ�P   <0��ArJ��   ArJ�P   ArMP   <'*ArM:�   ArMP   ArO�P   <�ArO��   ArO�P   ArQ�P   <ǰArR�   ArQ�P   ArThP   <@�bArT��   ArThP   ArV�P   <E(ArV��   ArV�P   ArYJP   <Ar�ArYo�   ArYJP   Ar[�P   <7�Ar[��   Ar[�P   Ar^,P   <1��Ar^Q�   Ar^,P   Ar`�P   <'��Ar`��   Ar`�P   ArcP   <5��Arc3�   ArcP   AreP   <:��Are��   AreP   Arg�P   <A�WArh�   Arg�P   ArjaP   <M�7Arj��   ArjaP   Arl�P   <Z|�Arl��   Arl�P   AroCP   <c8�Aroh�   AroCP   Arq�P   <d+�Arq��   Arq�P   Art%P   <ni1ArtJ�   Art%P   Arv�P   <{�;Arv��   Arv�P   AryP   <���Ary,�   AryP   Ar{xP   <��Ar{��   Ar{xP   Ar}�P   <��sAr~�   Ar}�P   Ar�ZP   <���Ar��   Ar�ZP   Ar��P   <��Ar���   Ar��P   Ar�<P   <���Ar�a�   Ar�<P   Ar��P   <�W}Ar���   Ar��P   Ar�P   <�ǟAr�C�   Ar�P   Ar��P   <��)Ar���   Ar��P   Ar� P   <���Ar�%�   Ar� P   Ar�qP   <�;wAr���   Ar�qP   Ar��P   <��*Ar��   Ar��P   Ar�SP   <��Ar�x�   Ar�SP   Ar��P   <�3eAr���   Ar��P   Ar�5P   <���Ar�Z�   Ar�5P   Ar��P   <�3�Ar���   Ar��P   Ar�P   <���Ar�<�   Ar�P   Ar��P   <���Ar���   Ar��P   Ar��P   <úAr��   Ar��P   Ar�jP   <��Ar���   Ar�jP   Ar��P   <�yAr� �   Ar��P   Ar�LP   <�05Ar�q�   Ar�LP   Ar��P   <�p�Ar���   Ar��P   Ar�.P   <��7Ar�S�   Ar�.P   Ar��P   <���Ar���   Ar��P   Ar�P   <��Ar�5�   Ar�P   Ar��P   <�'wAr���   Ar��P   Ar��P   <��mAr��   Ar��P   Ar�cP   <��FAr���   Ar�cP   Ar��P   <�Y�Ar���   Ar��P   Ar�EP   <��5Ar�j�   Ar�EP   ArĶP   <�cAr���   ArĶP   Ar�'P   <�k�Ar�L�   Ar�'P   ArɘP   <��Arɽ�   ArɘP   Ar�	P   <�B�Ar�.�   Ar�	P   Ar�zP   <��CArΟ�   Ar�zP   Ar��P   <��`Ar��   Ar��P   Ar�\P   <퓷ArӁ�   Ar�\P   Ar��P   <�=IAr���   Ar��P   Ar�>P   <�FAr�c�   Ar�>P   ArگP   <���Ar���   ArگP   Ar� P   <�7�Ar�E�   Ar� P   ArߑP   <��6Ar߶�   ArߑP   Ar�P   <��Ar�'�   Ar�P   Ar�sP   <�vAr��   Ar�sP   Ar��P   = 
Ar�	�   Ar��P   Ar�UP   =	��Ar�z�   Ar�UP   Ar��P   =�SAr���   Ar��P   Ar�7P   =��Ar�\�   Ar�7P   Ar�P   <�Ar���   Ar�P   Ar�P   =A�Ar�>�   Ar�P   Ar��P   =2IAr���   Ar��P   Ar��P   =�Ar� �   Ar��P   Ar�lP   =�SAr���   Ar�lP   Ar��P   =�vAr��   Ar��P   Ar�NP   =}(Ar�s�   Ar�NP   As�P   =#As��   As�P   As0P   =Q�AsU�   As0P   As�P   =��As��   As�P   As	P   =HAs	7�   As	P   As�P   =��As��   As�P   As�P   =(˰As�   As�P   AseP   =0��As��   AseP   As�P   =7r�As��   As�P   AsGP   =7�VAsl�   AsGP   As�P   =;[�As��   As�P   As)P   =B�AsN�   As)P   As�P   =HW�As��   As�P   AsP   =EN�As0�   AsP   As!|P   =Fa6As!��   As!|P   As#�P   =I��As$�   As#�P   As&^P   =QوAs&��   As&^P   As(�P   =WXAs(��   As(�P   As+@P   =S�OAs+e�   As+@P   As-�P   =V�:As-��   As-�P   As0"P   =b�tAs0G�   As0"P   As2�P   =mN+As2��   As2�P   As5P   =o�NAs5)�   As5P   As7uP   =h�!As7��   As7uP   As9�P   =kd�As:�   As9�P   As<WP   =vt�As<|�   As<WP   As>�P   =�RAs>��   As>�P   AsA9P   =�m�AsA^�   AsA9P   AsC�P   =���AsC��   AsC�P   AsFP   =�c�AsF@�   AsFP   AsH�P   =�VyAsH��   AsH�P   AsJ�P   =�]AsK"�   AsJ�P   AsMnP   =���AsM��   AsMnP   AsO�P   =��AsP�   AsO�P   AsRPP   =�SAsRu�   AsRPP   AsT�P   =��AsT��   AsT�P   AsW2P   =��AsWW�   AsW2P   AsY�P   =���AsY��   AsY�P   As\P   =�v�As\9�   As\P   As^�P   =�MAs^��   As^�P   As`�P   =���Asa�   As`�P   AscgP   =�m�Asc��   AscgP   Ase�P   =���Ase��   Ase�P   AshIP   =�� Ashn�   AshIP   Asj�P   =�n�Asj��   Asj�P   Asm+P   =�U#AsmP�   Asm+P   Aso�P   =�wAso��   Aso�P   AsrP   =�rGAsr2�   AsrP   Ast~P   =��9Ast��   Ast~P   Asv�P   =�g�Asw�   Asv�P   Asy`P   =�[%Asy��   Asy`P   As{�P   =�X�As{��   As{�P   As~BP   =�VAs~g�   As~BP   As��P   >�As���   As��P   As�$P   >Q�As�I�   As�$P   As��P   >&�As���   As��P   As�P   >U As�+�   As�P   As�wP   >�As���   As�wP   As��P   >#�As��   As��P   As�YP   >DBAs�~�   As�YP   As��P   >[vAs���   As��P   As�;P   >mAs�`�   As�;P   As��P   >F�As���   As��P   As�P   >�FAs�B�   As�P   As��P   >!�	As���   As��P   As��P   >$EAs�$�   As��P   As�pP   >&�As���   As�pP   As��P   >)}LAs��   As��P   As�RP   >.��As�w�   As�RP   As��P   >4n"As���   As��P   As�4P   >9�fAs�Y�   As�4P   As��P   >=rUAs���   As��P   As�P   >>�As�;�   As�P   As��P   >?��As���   As��P   As��P   >D#�As��   As��P   As�iP   >IʹAs���   As�iP   As��P   >N�As���   As��P   As�KP   >Ra�As�p�   As�KP   As��P   >W��As���   As��P   As�-P   >Z�As�R�   As�-P   AsP   >]�As���   AsP   As�P   >dHAs�4�   As�P   AsǀP   >h�NAsǥ�   AsǀP   As��P   >l��As��   As��P   As�bP   >n��Aṡ�   As�bP   As��P   >qnAs���   As��P   As�DP   >w�As�i�   As�DP   AsӵP   >}��As���   AsӵP   As�&P   >�kAs�K�   As�&P   AsؗP   >�o1Asؼ�   AsؗP   As�P   >���As�-�   As�P   As�yP   >��PAsݞ�   As�yP   As��P   >�As��   As��P   As�[P   >��As��   As�[P   As��P   >���As���   As��P   As�=P   >�D%As�b�   As�=P   As�P   >��vAs���   As�P   As�P   >��As�D�   As�P   As�P   >�^�As��   As�P   As�P   >���As�&�   As�P   As�rP   >�Y�As��   As�rP   As��P   >�Z�As��   As��P   As�TP   >�s�As�y�   As�TP   As��P   >��9As���   As��P   As�6P   >���As�[�   As�6P   As��P   >���As���   As��P   AtP   >�2yAt=�   AtP   At�P   >�yAt��   At�P   At�P   >��At�   At�P   At	kP   >���