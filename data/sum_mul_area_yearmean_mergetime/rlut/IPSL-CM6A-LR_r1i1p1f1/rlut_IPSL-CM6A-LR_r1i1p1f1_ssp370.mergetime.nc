CDF   �   
      time       bnds      lon       lat          5   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       4IPSL-CM6A-LR (2017):  atmos: LMDZ (NPv6, N96; 144 x 143 longitude/latitude; 79 levels; top level 40000 m) land: ORCHIDEE (v2.0, Water/Carbon/Energy mode) ocean: NEMO-OPA (eORCA1.3, tripolar primarily 1deg; 362 x 332 longitude/latitude; 75 levels; top grid cell 0-2 m) ocnBgchem: NEMO-PISCES seaIce: NEMO-LIM3   institution       2Institut Pierre Simon Laplace, Paris 75252, France     creation_date         2018-07-11T07:36:41Z   tracking_id       1hdl:21.14100/6e36e7b1-2ab1-470b-830e-5e8553ebfedd      description       CMIP6 historical   title         >IPSL-CM6A-LR model output prepared for CMIP6 / CMIP historical     activity_id       CMIP   contact       ipsl-cmip6@listes.ipsl.fr      data_specs_version        01.00.21   dr2xml_version        1.11   experiment_id         
historical     
experiment        )all-forcing simulation of the recent past      external_variables        	areacella      forcing_index               	frequency         year   grid      	LMDZ grid      
grid_label        gr     nominal_resolution        250 km     initialization_index            institution_id        IPSL   license      ICMIP6 model data produced by IPSL is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://cmc.ipsl.fr/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      mip_era       CMIP6      parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      IPSL-CM6A-LR   parent_time_units         days since 1850-01-01 00:00:00     branch_method         standard   branch_time_in_parent         @�f�       branch_time_in_child                 physics_index               product       model-output   realm         atmos      	source_id         IPSL-CM6A-LR   source_type       	AOGCM BGC      sub_experiment_id         none   sub_experiment        none   table_id      Amon   variable_id       rlut   EXPID         
historical     CMIP6_CV_version      cv=6.2.3.5-2-g63b123e      dr2xml_md5sum          f1e40c1fc5d8281f865f72fbf4e38f9d   model_version         6.1.5      parent_variant_label      r1i1p1f1   name      �/ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_%start_date%-%end_date%   further_info_url      Ohttps://furtherinfo.es-doc.org/CMIP6.IPSL.IPSL-CM6A-LR.historical.none.r1i1p1f1    variant_label         r1i1p1f1   realization_index               history      �Tue May 30 16:58:56 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/IPSL-CM6A-LR_r1i1p1f1/rlut_IPSL-CM6A-LR_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/IPSL-CM6A-LR_r1i1p1f1/CMIP6.ScenarioMIP.IPSL.IPSL-CM6A-LR.ssp370.r1i1p1f1.Amon.rlut.gr.v20190119/rlut_Amon_IPSL-CM6A-LR_ssp370_r1i1p1f1_gr_201501-210012.yearmean.mul.areacella_ssp370_v20190119.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/IPSL-CM6A-LR_r1i1p1f1/rlut_IPSL-CM6A-LR_r1i1p1f1_ssp370.mergetime.nc
Tue May 30 16:58:55 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/IPSL-CM6A-LR_r1i1p1f1/CMIP6.CMIP.IPSL.IPSL-CM6A-LR.historical.r1i1p1f1.Amon.rlut.gr.v20180803/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.yearmean.mul.areacella_historical_v20180803.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/IPSL-CM6A-LR_r1i1p1f1/rlut_IPSL-CM6A-LR_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 03:56:18 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/IPSL-CM6A-LR_r1i1p1f1/CMIP6.CMIP.IPSL.IPSL-CM6A-LR.historical.r1i1p1f1.Amon.rlut.gr.v20180803/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/IPSL-CM6A-LR_r1i1p1f1/CMIP6.CMIP.IPSL.IPSL-CM6A-LR.historical.r1i1p1f1.fx.areacella.gr.v20180803/areacella_fx_IPSL-CM6A-LR_historical_r1i1p1f1_gr.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/IPSL-CM6A-LR_r1i1p1f1/CMIP6.CMIP.IPSL.IPSL-CM6A-LR.historical.r1i1p1f1.Amon.rlut.gr.v20180803/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.yearmean.mul.areacella_historical_v20180803.fldsum.nc
Sat Dec  1 12:12:43 2018: ncatted -O -a realization_index,global,m,i,1 /ccc/work/cont003/cmip6/cmip6/onhold/CM61-LR-histEXT-03.1910/files+ext/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc
Sat Dec  1 12:04:10 2018: ncatted -O -a realization_index,global,m,i,1 /ccc/work/cont003/cmip6/cmip6/onhold/CM61-LR-hist-03.1910/files/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc
Sat Dec  1 10:59:40 2018: ncatted -O -a realization_index,global,m,i,1 /ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc
Fri Nov 30 16:48:40 2018: ncatted -O -a realization_index,global,m,s,1 /ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc
Thu Nov 29 16:57:05 2018: ncatted -O -a variant_label,global,m,c,r1i1p1f1 /ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r3i1p1f1_gr_185001-201412.nc
Thu Nov 29 16:57:05 2018: ncatted -O -a further_info_url,global,m,c,https://furtherinfo.es-doc.org/CMIP6.IPSL.IPSL-CM6A-LR.historical.none.r1i1p1f1 /ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r3i1p1f1_gr_185001-201412.nc
Thu Nov 29 16:57:05 2018: ncatted -O -a name,global,m,c,/ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_%start_date%-%end_date% /ccc/work/cont003/gencmip6/p86caub/IGCM_OUT/IPSLCM6/PROD/historical/CM61-LR-hist-03.1910/CMIP6/ATM/rlut_Amon_IPSL-CM6A-LR_historical_r3i1p1f1_gr_185001-201412.nc
Mon Sep  3 14:51:42 2018: ncatted -O -a parent_variant_label,global,m,c,r1i1p1f1 rlut_Amon_IPSL-CM6A-LR_historical_r3i1p1f1_gr_185001-201412.nc
none     NCO       "4.6.0"    CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               !�   	time_bnds                                 !�   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               !�   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               !�   rlut                      standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   online_operation      average    interval_operation        900 s      interval_write        1 month    description       Iat the top of the atmosphere (to be compared with satellite measurements)      positive      up     history       none   cell_measures         area: areacella             !�                Aq���   Aq��P   Aq�P   [�e1Aq�6�   Aq�P   Aq��P   [׉�Aq���   Aq��P   Aq��P   [׽2Aq��   Aq��P   Aq�dP   [��rAq���   Aq�dP   Aq��P   [׽�Aq���   Aq��P   Aq�FP   [���Aq�k�   Aq�FP   Aq��P   [׹4Aq���   Aq��P   Aq�(P   [��zAq�M�   Aq�(P   Aq��P   [��Aq���   Aq��P   Aq�
P   [���Aq�/�   Aq�
P   Aq�{P   [���Aq���   Aq�{P   Aq��P   [�� Aq��   Aq��P   Aq�]P   [בXAqĂ�   Aq�]P   Aq��P   [�d�Aq���   Aq��P   Aq�?P   [ט�Aq�d�   Aq�?P   Aq˰P   [�ՆAq���   Aq˰P   Aq�!P   [�
�Aq�F�   Aq�!P   AqВP   [��Aqз�   AqВP   Aq�P   [��Aq�(�   Aq�P   Aq�tP   [׈�Aqՙ�   Aq�tP   Aq��P   [׹Aq�
�   Aq��P   Aq�VP   [���Aq�{�   Aq�VP   Aq��P   [��Aq���   Aq��P   Aq�8P   [� Aq�]�   Aq�8P   Aq�P   [���Aq���   Aq�P   Aq�P   [ׯ�Aq�?�   Aq�P   Aq�P   [��oAq��   Aq�P   Aq��P   [��Aq�!�   Aq��P   Aq�mP   [��pAq��   Aq�mP   Aq��P   [נdAq��   Aq��P   Aq�OP   [�}�Aq�t�   Aq�OP   Aq��P   [׆Aq���   Aq��P   Aq�1P   [�Q�Aq�V�   Aq�1P   Aq��P   [��Aq���   Aq��P   Aq�P   [��KAq�8�   Aq�P   Aq��P   [�~Aq���   Aq��P   Aq��P   [�6�Aq��   Aq��P   ArfP   [�B�Ar��   ArfP   Ar�P   [�?hAr��   Ar�P   ArHP   [�=�Arm�   ArHP   Ar�P   [�{Ar��   Ar�P   Ar*P   [�UfArO�   Ar*P   Ar�P   [׎�Ar��   Ar�P   ArP   [�j�Ar1�   ArP   Ar}P   [�iFAr��   Ar}P   Ar�P   [���Ar�   Ar�P   Ar_P   [ז�Ar��   Ar_P   Ar�P   [ׁjAr��   Ar�P   ArAP   [�oeArf�   ArAP   Ar�P   [ד�Ar��   Ar�P   Ar!#P   [��Ar!H�   Ar!#P   Ar#�P   [׸�Ar#��   Ar#�P   Ar&P   [�sAr&*�   Ar&P   Ar(vP   [�v�Ar(��   Ar(vP   Ar*�P   [�q9Ar+�   Ar*�P   Ar-XP   [�Q�Ar-}�   Ar-XP   Ar/�P   [�Q�Ar/��   Ar/�P   Ar2:P   [�O&Ar2_�   Ar2:P   Ar4�P   [�pAr4��   Ar4�P   Ar7P   [קAr7A�   Ar7P   Ar9�P   [ב�Ar9��   Ar9�P   Ar;�P   [אqAr<#�   Ar;�P   Ar>oP   [�^|Ar>��   Ar>oP   Ar@�P   [�$�ArA�   Ar@�P   ArCQP   [�3kArCv�   ArCQP   ArE�P   [�VmArE��   ArE�P   ArH3P   [׿�ArHX�   ArH3P   ArJ�P   [מCArJ��   ArJ�P   ArMP   [�[VArM:�   ArMP   ArO�P   [�|(ArO��   ArO�P   ArQ�P   [�gHArR�   ArQ�P   ArThP   [א<ArT��   ArThP   ArV�P   [�J�ArV��   ArV�P   ArYJP   [�!?ArYo�   ArYJP   Ar[�P   [�l�Ar[��   Ar[�P   Ar^,P   [�r{Ar^Q�   Ar^,P   Ar`�P   [�^�Ar`��   Ar`�P   ArcP   [�~SArc3�   ArcP   AreP   [�o�Are��   AreP   Arg�P   [�T�Arh�   Arg�P   ArjaP   [�@�Arj��   ArjaP   Arl�P   [�e7Arl��   Arl�P   AroCP   [�P*Aroh�   AroCP   Arq�P   [׋�Arq��   Arq�P   Art%P   [ג�ArtJ�   Art%P   Arv�P   [ׅ�Arv��   Arv�P   AryP   [נcAry,�   AryP   Ar{xP   [חAr{��   Ar{xP   Ar}�P   [�߹Ar~�   Ar}�P   Ar�ZP   [ג�Ar��   Ar�ZP   Ar��P   [לAr���   Ar��P   Ar�<P   [�sAr�a�   Ar�<P   Ar��P   [��YAr���   Ar��P   Ar�P   [��CAr�C�   Ar�P   Ar��P   [�� Ar���   Ar��P   Ar� P   [׶�Ar�%�   Ar� P   Ar�qP   [��9Ar���   Ar�qP   Ar��P   [�� Ar��   Ar��P   Ar�SP   [���Ar�x�   Ar�SP   Ar��P   [���Ar���   Ar��P   Ar�5P   [׶�Ar�Z�   Ar�5P   Ar��P   [��iAr���   Ar��P   Ar�P   [��\Ar�<�   Ar�P   Ar��P   [��Ar���   Ar��P   Ar��P   [��]Ar��   Ar��P   Ar�jP   [חvAr���   Ar�jP   Ar��P   [�/Ar� �   Ar��P   Ar�LP   [�9CAr�q�   Ar�LP   Ar��P   [�Ar���   Ar��P   Ar�.P   [׺Ar�S�   Ar�.P   Ar��P   [��!Ar���   Ar��P   Ar�P   [״�Ar�5�   Ar�P   Ar��P   [�ӨAr���   Ar��P   Ar��P   [��=Ar��   Ar��P   Ar�cP   [�[qAr���   Ar�cP   Ar��P   [ט�Ar���   Ar��P   Ar�EP   [׷Ar�j�   Ar�EP   ArĶP   [װAr���   ArĶP   Ar�'P   [ׂBAr�L�   Ar�'P   ArɘP   [פ�Arɽ�   ArɘP   Ar�	P   [�|�Ar�.�   Ar�	P   Ar�zP   [�r�ArΟ�   Ar�zP   Ar��P   [׆�Ar��   Ar��P   Ar�\P   [נQArӁ�   Ar�\P   Ar��P   [�gcAr���   Ar��P   Ar�>P   [�Z@Ar�c�   Ar�>P   ArگP   [�X[Ar���   ArگP   Ar� P   [�fcAr�E�   Ar� P   ArߑP   [�N�Ar߶�   ArߑP   Ar�P   [�`\Ar�'�   Ar�P   Ar�sP   [�BAr��   Ar�sP   Ar��P   [�R�Ar�	�   Ar��P   Ar�UP   [��FAr�z�   Ar�UP   Ar��P   [֞�Ar���   Ar��P   Ar�7P   [��Ar�\�   Ar�7P   Ar�P   [���Ar���   Ar�P   Ar�P   [�XAr�>�   Ar�P   Ar��P   [�M�Ar���   Ar��P   Ar��P   [׏�Ar� �   Ar��P   Ar�lP   [���Ar���   Ar�lP   Ar��P   [�t�Ar��   Ar��P   Ar�NP   [���Ar�s�   Ar�NP   As�P   [֬�As��   As�P   As0P   [��9AsU�   As0P   As�P   [�UAs��   As�P   As	P   [�B;As	7�   As	P   As�P   [�A�As��   As�P   As�P   [�FFAs�   As�P   AseP   [�5�As��   AseP   As�P   [�B(As��   As�P   AsGP   [�B�Asl�   AsGP   As�P   [�k|As��   As�P   As)P   [��AsN�   As)P   As�P   [�H�As��   As�P   AsP   [�]rAs0�   AsP   As!|P   [�c�As!��   As!|P   As#�P   [�CgAs$�   As#�P   As&^P   [�A�As&��   As&^P   As(�P   [�)�As(��   As(�P   As+@P   [�xLAs+e�   As+@P   As-�P   [���As-��   As-�P   As0"P   [׊2As0G�   As0"P   As2�P   [לyAs2��   As2�P   As5P   [׆�As5)�   As5P   As7uP   [ע�As7��   As7uP   As9�P   [�x\As:�   As9�P   As<WP   [�ehAs<|�   As<WP   As>�P   [ש�As>��   As>�P   AsA9P   [ײQAsA^�   AsA9P   AsC�P   [ו�AsC��   AsC�P   AsFP   [�X�AsF@�   AsFP   AsH�P   [��AsH��   AsH�P   AsJ�P   [�z�AsK"�   AsJ�P   AsMnP   [�tHAsM��   AsMnP   AsO�P   [�b�AsP�   AsO�P   AsRPP   [ׁ�AsRu�   AsRPP   AsT�P   [׷�AsT��   AsT�P   AsW2P   [הcAsWW�   AsW2P   AsY�P   [׼jAsY��   AsY�P   As\P   [��5As\9�   As\P   As^�P   [ׂ�As^��   As^�P   As`�P   [�v�Asa�   As`�P   AscgP   [ײ�Asc��   AscgP   Ase�P   [מpAse��   Ase�P   AshIP   [�j)Ashn�   AshIP   Asj�P   [��gAsj��   Asj�P   Asm+P   [���AsmP�   Asm+P   Aso�P   [���Aso��   Aso�P   AsrP   [���Asr2�   AsrP   Ast~P   [��Ast��   Ast~P   Asv�P   [�Asw�   Asv�P   Asy`P   [�(�Asy��   Asy`P   As{�P   [�FAs{��   As{�P   As~BP   [��lAs~g�   As~BP   As��P   [��`As���   As��P   As�$P   [��As�I�   As�$P   As��P   [�9As���   As��P   As�P   [�BfAs�+�   As�P   As�wP   [�BAs���   As�wP   As��P   [�V�As��   As��P   As�YP   [�`As�~�   As�YP   As��P   [�|�As���   As��P   As�;P   [�h'As�`�   As�;P   As��P   [�\OAs���   As��P   As�P   [�jAs�B�   As�P   As��P   [�As���   As��P   As��P   [�X�As�$�   As��P   As�pP   [�8DAs���   As�pP   As��P   [؎UAs��   As��P   As�RP   [�yAs�w�   As�RP   As��P   [�<�As���   As��P   As�4P   [�NIAs�Y�   As�4P   As��P   [؝As���   As��P   As�P   [��BAs�;�   As�P   As��P   [؏{As���   As��P   As��P   [�^�As��   As��P   As�iP   [��As���   As�iP   As��P   [��|As���   As��P   As�KP   [ئ\As�p�   As�KP   As��P   [��]As���   As��P   As�-P   [ؐ^As�R�   As�-P   AsP   [خ�As���   AsP   As�P   [��As�4�   As�P   AsǀP   [��2Asǥ�   AsǀP   As��P   [��As��   As��P   As�bP   [��zAṡ�   As�bP   As��P   [�<tAs���   As��P   As�DP   [� As�i�   As�DP   AsӵP   [�.�As���   AsӵP   As�&P   [�D�As�K�   As�&P   AsؗP   [�|Asؼ�   AsؗP   As�P   [���As�-�   As�P   As�yP   [�y�Asݞ�   As�yP   As��P   [�t�As��   As��P   As�[P   [�0�As��   As�[P   As��P   [ٖ�As���   As��P   As�=P   [�z~As�b�   As�=P   As�P   [���As���   As�P   As�P   [�A�As�D�   As�P   As�P   [ّ�As��   As�P   As�P   [٬�As�&�   As�P   As�rP   [ًPAs��   As�rP   As��P   [٫As��   As��P   As�TP   [�:-As�y�   As�TP   As��P   [�4�As���   As��P   As�6P   [��yAs�[�   As�6P   As��P   [��AAs���   As��P   AtP   [ڃAt=�   AtP   At�P   [�2�At��   At�P   At�P   [�At�   At�P   At	kP   [�X