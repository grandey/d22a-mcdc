CDF   �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       0MIROC6 (2017): 
aerosol: SPRINTARS6.0
atmos: CCSR AGCM (T85; 256 x 128 longitude/latitude; 81 levels; top level 0.004 hPa)
atmosChem: none
land: MATSIRO6.0
landIce: none
ocean: COCO4.9 (tripolar primarily 1deg; 360 x 256 longitude/latitude; 63 levels; top grid cell 0-2 m)
ocnBgchem: none
seaIce: COCO4.9   institution      QJAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2018-11-30T16:10:21Z   data_specs_version        01.00.28   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Jhttps://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.historical.none.r1i1p1f1     grid      #native atmosphere T85 Gaussian grid    
grid_label        gn     history      Wed Nov 09 19:01:02 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/MIROC6_r1i1p1f1/rlut_MIROC6_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MIROC6_r1i1p1f1/CMIP6.ScenarioMIP.MIROC.MIROC6.ssp126.r1i1p1f1.Amon.rlut.gn.v20190627/rlut_Amon_MIROC6_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp126_v20190627.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/MIROC6_r1i1p1f1/rlut_MIROC6_r1i1p1f1_ssp126.mergetime.nc
Wed Nov 09 19:01:02 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rlut.gn.v20181212/rlut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.yearmean.mul.areacella_historical_v20190311.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rlut.gn.v20181212/rlut_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.yearmean.mul.areacella_historical_v20190311.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/MIROC6_r1i1p1f1/rlut_MIROC6_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 03:57:16 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rlut.gn.v20181212/rlut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.fx.areacella.gn.v20190311/areacella_fx_MIROC6_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rlut.gn.v20181212/rlut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.yearmean.mul.areacella_historical_v20190311.fldsum.nc
2018-11-30T16:10:21Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.   initialization_index            institution_id        MIROC      mip_era       CMIP6      nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      MIROC6     parent_time_units         days since 3200-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         MIROC6     source_type       	AOGCM AER      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(06 November 2018) MD5:0728c79344e0f262bb76e4f9ff0d9afc      title          MIROC6 output prepared for CMIP6   variable_id       rlut   variant_label         r1i1p1f1   license      !CMIP6 model data produced by MIROC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      cmor_version      3.3.2      tracking_id       1hdl:21.14100/94f21608-c53f-4b60-8289-ea7466aa6061      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                      standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         OLR    original_units        W/m**2     history       �2018-11-30T16:10:21Z altered by CMOR: Converted units from 'W/m**2' to 'W m-2'. 2018-11-30T16:10:21Z altered by CMOR: replaced missing value flag (-999) with standard missing value (1e+20). 2018-11-30T16:10:21Z altered by CMOR: Inverted axis: lat.    cell_measures         area: areacella                             Aq���   Aq��P   Aq�P   [Й/Aq�6�   Aq�P   Aq��P   [�n�Aq���   Aq��P   Aq��P   [�o3Aq��   Aq��P   Aq�dP   [�cLAq���   Aq�dP   Aq��P   [МKAq���   Aq��P   Aq�FP   [�lxAq�k�   Aq�FP   Aq��P   [�C�Aq���   Aq��P   Aq�(P   [�E�Aq�M�   Aq�(P   Aq��P   [ІBAq���   Aq��P   Aq�
P   [ЦAq�/�   Aq�
P   Aq�{P   [��[Aq���   Aq�{P   Aq��P   [�[�Aq��   Aq��P   Aq�]P   [�N}AqĂ�   Aq�]P   Aq��P   [��Aq���   Aq��P   Aq�?P   [�	Aq�d�   Aq�?P   Aq˰P   [��uAq���   Aq˰P   Aq�!P   [�-vAq�F�   Aq�!P   AqВP   [�0�Aqз�   AqВP   Aq�P   [�f-Aq�(�   Aq�P   Aq�tP   [���Aqՙ�   Aq�tP   Aq��P   [��CAq�
�   Aq��P   Aq�VP   [п�Aq�{�   Aq�VP   Aq��P   [ЧAq���   Aq��P   Aq�8P   [ж�Aq�]�   Aq�8P   Aq�P   [��Aq���   Aq�P   Aq�P   [И8Aq�?�   Aq�P   Aq�P   [о�Aq��   Aq�P   Aq��P   [�\,Aq�!�   Aq��P   Aq�mP   [�/�Aq��   Aq�mP   Aq��P   [Л=Aq��   Aq��P   Aq�OP   [ЖbAq�t�   Aq�OP   Aq��P   [ЕAq���   Aq��P   Aq�1P   [�^QAq�V�   Aq�1P   Aq��P   [Э
Aq���   Aq��P   Aq�P   [�q�Aq�8�   Aq�P   Aq��P   [�ܒAq���   Aq��P   Aq��P   [�*CAq��   Aq��P   ArfP   [�$�Ar��   ArfP   Ar�P   [ЕDAr��   Ar�P   ArHP   [�nRArm�   ArHP   Ar�P   [��Ar��   Ar�P   Ar*P   [�~`ArO�   Ar*P   Ar�P   [�W�Ar��   Ar�P   ArP   [�W_Ar1�   ArP   Ar}P   [НAr��   Ar}P   Ar�P   [В�Ar�   Ar�P   Ar_P   [�;�Ar��   Ar_P   Ar�P   [�kNAr��   Ar�P   ArAP   [МArf�   ArAP   Ar�P   [�e�Ar��   Ar�P   Ar!#P   [м�Ar!H�   Ar!#P   Ar#�P   [мAr#��   Ar#�P   Ar&P   [�_�Ar&*�   Ar&P   Ar(vP   [�yAr(��   Ar(vP   Ar*�P   [ϪdAr+�   Ar*�P   Ar-XP   [�� Ar-}�   Ar-XP   Ar/�P   [�%Ar/��   Ar/�P   Ar2:P   [�.5Ar2_�   Ar2:P   Ar4�P   [�*�Ar4��   Ar4�P   Ar7P   [лAAr7A�   Ar7P   Ar9�P   [Ж�Ar9��   Ar9�P   Ar;�P   [��Ar<#�   Ar;�P   Ar>oP   [��BAr>��   Ar>oP   Ar@�P   [���ArA�   Ar@�P   ArCQP   [�7�ArCv�   ArCQP   ArE�P   [��ArE��   ArE�P   ArH3P   [�7�ArHX�   ArH3P   ArJ�P   [Ћ�ArJ��   ArJ�P   ArMP   [��*ArM:�   ArMP   ArO�P   [�#�ArO��   ArO�P   ArQ�P   [ϼ�ArR�   ArQ�P   ArThP   [��ArT��   ArThP   ArV�P   [�l�ArV��   ArV�P   ArYJP   [�4�ArYo�   ArYJP   Ar[�P   [�v!Ar[��   Ar[�P   Ar^,P   [БAr^Q�   Ar^,P   Ar`�P   [�d�Ar`��   Ar`�P   ArcP   [�Arc3�   ArcP   AreP   [�c�Are��   AreP   Arg�P   [��Arh�   Arg�P   ArjaP   [��Arj��   ArjaP   Arl�P   [�,sArl��   Arl�P   AroCP   [�>6Aroh�   AroCP   Arq�P   [� �Arq��   Arq�P   Art%P   [�f�ArtJ�   Art%P   Arv�P   [�P�Arv��   Arv�P   AryP   [�V,Ary,�   AryP   Ar{xP   [��LAr{��   Ar{xP   Ar}�P   [В�Ar~�   Ar}�P   Ar�ZP   [�BAr��   Ar�ZP   Ar��P   [��sAr���   Ar��P   Ar�<P   [���Ar�a�   Ar�<P   Ar��P   [�AfAr���   Ar��P   Ar�P   [�jAr�C�   Ar�P   Ar��P   [ЧYAr���   Ar��P   Ar� P   [�ɣAr�%�   Ar� P   Ar�qP   [�I�Ar���   Ar�qP   Ar��P   [�؛Ar��   Ar��P   Ar�SP   [�G�Ar�x�   Ar�SP   Ar��P   [�X�Ar���   Ar��P   Ar�5P   [�y�Ar�Z�   Ar�5P   Ar��P   [�7�Ar���   Ar��P   Ar�P   [�Ar�<�   Ar�P   Ar��P   [��Ar���   Ar��P   Ar��P   [��"Ar��   Ar��P   Ar�jP   [�,�Ar���   Ar�jP   Ar��P   [Ќ�Ar� �   Ar��P   Ar�LP   [КyAr�q�   Ar�LP   Ar��P   [Љ�Ar���   Ar��P   Ar�.P   [�uAr�S�   Ar�.P   Ar��P   [��Ar���   Ar��P   Ar�P   [ϵ�Ar�5�   Ar�P   Ar��P   [��Ar���   Ar��P   Ar��P   [��Ar��   Ar��P   Ar�cP   [�S�Ar���   Ar�cP   Ar��P   [��Ar���   Ar��P   Ar�EP   [�PAr�j�   Ar�EP   ArĶP   [�r.Ar���   ArĶP   Ar�'P   [�	bAr�L�   Ar�'P   ArɘP   [ϕ>Arɽ�   ArɘP   Ar�	P   [�I_Ar�.�   Ar�	P   Ar�zP   [Ϟ	ArΟ�   Ar�zP   Ar��P   [�_�Ar��   Ar��P   Ar�\P   [φ�ArӁ�   Ar�\P   Ar��P   [�uAr���   Ar��P   Ar�>P   [���Ar�c�   Ar�>P   ArگP   [ϕtAr���   ArگP   Ar� P   [�9�Ar�E�   Ar� P   ArߑP   [ϥ�Ar߶�   ArߑP   Ar�P   [��Ar�'�   Ar�P   Ar�sP   [�EwAr��   Ar�sP   Ar��P   [�hAr�	�   Ar��P   Ar�UP   [�nUAr�z�   Ar�UP   Ar��P   [�uAr���   Ar��P   Ar�7P   [ϸ�Ar�\�   Ar�7P   Ar�P   [ϱ�Ar���   Ar�P   Ar�P   [�	�Ar�>�   Ar�P   Ar��P   [�A�Ar���   Ar��P   Ar��P   [�oQAr� �   Ar��P   Ar�lP   [�6NAr���   Ar�lP   Ar��P   [�V�Ar��   Ar��P   Ar�NP   [�Ar�s�   Ar�NP   As�P   [���As��   As�P   As0P   [ν)AsU�   As0P   As�P   [�*LAs��   As�P   As	P   [α�As	7�   As	P   As�P   [�ImAs��   As�P   As�P   [���As�   As�P   AseP   [��As��   AseP   As�P   [Ϭ�As��   As�P   AsGP   [��Asl�   AsGP   As�P   [�)yAs��   As�P   As)P   [�0�AsN�   As)P   As�P   [�K	As��   As�P   AsP   [�ոAs0�   AsP   As!|P   [�k�As!��   As!|P   As#�P   [�vAs$�   As#�P   As&^P   [�	�As&��   As&^P   As(�P   [ω2As(��   As(�P   As+@P   [��9As+e�   As+@P   As-�P   [��As-��   As-�P   As0"P   [��As0G�   As0"P   As2�P   [�*�As2��   As2�P   As5P   [��AAs5)�   As5P   As7uP   [���As7��   As7uP   As9�P   [��
As:�   As9�P   As<WP   [Ν:As<|�   As<WP   As>�P   [μAs>��   As>�P   AsA9P   [�9HAsA^�   AsA9P   AsC�P   [ύ4AsC��   AsC�P   AsFP   [��AsF@�   AsFP   AsH�P   [�LAsH��   AsH�P   AsJ�P   [��AsK"�   AsJ�P   AsMnP   [Ϝ<AsM��   AsMnP   AsO�P   [�AsP�   AsO�P   AsRPP   [�[�AsRu�   AsRPP   AsT�P   [��aAsT��   AsT�P   AsW2P   [�bOAsWW�   AsW2P   AsY�P   [Ϧ6AsY��   AsY�P   As\P   [�қAs\9�   As\P   As^�P   [�rzAs^��   As^�P   As`�P   [ЊAsa�   As`�P   AscgP   [ϲAAsc��   AscgP   Ase�P   [�0�Ase��   Ase�P   AshIP   [���Ashn�   AshIP   Asj�P   [�gAsj��   Asj�P   Asm+P   [�M�AsmP�   Asm+P   Aso�P   [���Aso��   Aso�P   AsrP   [�n�Asr2�   AsrP   Ast~P   [���Ast��   Ast~P   Asv�P   [��[Asw�   Asv�P   Asy`P   [��SAsy��   Asy`P   As{�P   [��As{��   As{�P   As~BP   [СAs~g�   As~BP   As��P   [�FyAs���   As��P   As�$P   [�,UAs�I�   As�$P   As��P   [�7As���   As��P   As�P   [ЂmAs�+�   As�P   As�wP   [�9�As���   As�wP   As��P   [�]�As��   As��P   As�YP   [Ќ�As�~�   As�YP   As��P   [�5 As���   As��P   As�;P   [�,\As�`�   As�;P   As��P   [�0}As���   As��P   As�P   [�6}As�B�   As�P   As��P   [М[As���   As��P   As��P   [��pAs�$�   As��P   As�pP   [��As���   As�pP   As��P   [��As��   As��P   As�RP   [ШAs�w�   As�RP   As��P   [��<As���   As��P   As�4P   [аNAs�Y�   As�4P   As��P   [�o�As���   As��P   As�P   [йAs�;�   As�P   As��P   [�rBAs���   As��P   As��P   [�b(As��   As��P   As�iP   [СAs���   As�iP   As��P   [�O�As���   As��P   As�KP   [��As�p�   As�KP   As��P   [�y�As���   As��P   As�-P   [��"As�R�   As�-P   AsP   [���As���   AsP   As�P   [�gAs�4�   As�P   AsǀP   [�-Asǥ�   AsǀP   As��P   [��BAs��   As��P   As�bP   [ХgAṡ�   As�bP   As��P   [�˭As���   As��P   As�DP   [���As�i�   As�DP   AsӵP   [�K�As���   AsӵP   As�&P   [�ǒAs�K�   As�&P   AsؗP   [РlAsؼ�   AsؗP   As�P   [�'As�-�   As�P   As�yP   [��Asݞ�   As�yP   As��P   [�0As��   As��P   As�[P   [�/As��   As�[P   As��P   [��As���   As��P   As�=P   [Ѡ�As�b�   As�=P   As�P   [�:As���   As�P   As�P   [Б�As�D�   As�P   As�P   [н�As��   As�P   As�P   [�As�&�   As�P   As�rP   [��As��   As�rP   As��P   [��As��   As��P   As�TP   [�V!As�y�   As�TP   As��P   [���As���   As��P   As�6P   [б|As�[�   As�6P   As��P   [м�As���   As��P   AtP   [���At=�   AtP   At�P   [�/]At��   At�P   At�P   [ѩ�At�   At�P   At	kP   [��L