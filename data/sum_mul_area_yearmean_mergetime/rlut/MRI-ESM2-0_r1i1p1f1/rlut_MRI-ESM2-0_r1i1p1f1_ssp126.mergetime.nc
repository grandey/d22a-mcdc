CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-02-20T02:45:37Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.historical.none.r1i1p1f1   grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      �Tue May 30 16:58:59 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_201501-210012.yearmean.mul.areacella_ssp126_v20190603.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_210101-230012.yearmean.mul.areacella_ssp126_v20190603.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_ssp126.mergetime.nc
Tue May 30 16:58:59 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190603.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 04:04:23 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.fx.areacella.gn.v20190603/areacella_fx_MRI-ESM2-0_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190603.fldsum.nc
2019-02-20T02:45:37Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_historical_101 (sfc_avr_mon.ctl)    initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rlut   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/424f453d-9b80-4d60-9b9d-ec646b7cf434      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                    lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                   
   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         ULWT   cell_measures         area: areacella    history       r2019-02-20T02:45:37Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).                              Aq���   Aq��P   Aq�P   [�3uAq�6�   Aq�P   Aq��P   [�H6Aq���   Aq��P   Aq��P   [��Aq��   Aq��P   Aq�dP   [گOAq���   Aq�dP   Aq��P   [��Aq���   Aq��P   Aq�FP   [�#Aq�k�   Aq�FP   Aq��P   [��Aq���   Aq��P   Aq�(P   [ڿ[Aq�M�   Aq�(P   Aq��P   [���Aq���   Aq��P   Aq�
P   [�7�Aq�/�   Aq�
P   Aq�{P   [��(Aq���   Aq�{P   Aq��P   [ڭiAq��   Aq��P   Aq�]P   [�|AqĂ�   Aq�]P   Aq��P   [���Aq���   Aq��P   Aq�?P   [ڴAq�d�   Aq�?P   Aq˰P   [ڷzAq���   Aq˰P   Aq�!P   [گWAq�F�   Aq�!P   AqВP   [ڰaAqз�   AqВP   Aq�P   [�aAq�(�   Aq�P   Aq�tP   [ڴ�Aqՙ�   Aq�tP   Aq��P   [ڑ�Aq�
�   Aq��P   Aq�VP   [��Aq�{�   Aq�VP   Aq��P   [ڄ�Aq���   Aq��P   Aq�8P   [�'Aq�]�   Aq�8P   Aq�P   [�O�Aq���   Aq�P   Aq�P   [��Aq�?�   Aq�P   Aq�P   [ڨtAq��   Aq�P   Aq��P   [ڎ�Aq�!�   Aq��P   Aq�mP   [��hAq��   Aq�mP   Aq��P   [ڬAq��   Aq��P   Aq�OP   [ړ1Aq�t�   Aq�OP   Aq��P   [��bAq���   Aq��P   Aq�1P   [��Aq�V�   Aq�1P   Aq��P   [�<Aq���   Aq��P   Aq�P   [ٜ�Aq�8�   Aq�P   Aq��P   [��nAq���   Aq��P   Aq��P   [�NjAq��   Aq��P   ArfP   [�C�Ar��   ArfP   Ar�P   [ڄ4Ar��   Ar�P   ArHP   [�ipArm�   ArHP   Ar�P   [�Q�Ar��   Ar�P   Ar*P   [ڬqArO�   Ar*P   Ar�P   [ځ�Ar��   Ar�P   ArP   [ڽ Ar1�   ArP   Ar}P   [��.Ar��   Ar}P   Ar�P   [���Ar�   Ar�P   Ar_P   [��Ar��   Ar_P   Ar�P   [ڻ�Ar��   Ar�P   ArAP   [��Arf�   ArAP   Ar�P   [�ʕAr��   Ar�P   Ar!#P   [�8�Ar!H�   Ar!#P   Ar#�P   [چ?Ar#��   Ar#�P   Ar&P   [ڤ;Ar&*�   Ar&P   Ar(vP   [���Ar(��   Ar(vP   Ar*�P   [�:;Ar+�   Ar*�P   Ar-XP   [��_Ar-}�   Ar-XP   Ar/�P   [�saAr/��   Ar/�P   Ar2:P   [�`Ar2_�   Ar2:P   Ar4�P   [ڛ�Ar4��   Ar4�P   Ar7P   [ڤ�Ar7A�   Ar7P   Ar9�P   [��QAr9��   Ar9�P   Ar;�P   [�,[Ar<#�   Ar;�P   Ar>oP   [�
Ar>��   Ar>oP   Ar@�P   [�;ArA�   Ar@�P   ArCQP   [�g�ArCv�   ArCQP   ArE�P   [ڤ�ArE��   ArE�P   ArH3P   [�UPArHX�   ArH3P   ArJ�P   [�8�ArJ��   ArJ�P   ArMP   [�P�ArM:�   ArMP   ArO�P   [�)ArO��   ArO�P   ArQ�P   [�*\ArR�   ArQ�P   ArThP   [�.�ArT��   ArThP   ArV�P   [�S%ArV��   ArV�P   ArYJP   [ڢdArYo�   ArYJP   Ar[�P   [���Ar[��   Ar[�P   Ar^,P   [��Ar^Q�   Ar^,P   Ar`�P   [�;�Ar`��   Ar`�P   ArcP   [�<IArc3�   ArcP   AreP   [�:�Are��   AreP   Arg�P   [�w�Arh�   Arg�P   ArjaP   [�E�Arj��   ArjaP   Arl�P   [�	�Arl��   Arl�P   AroCP   [��Aroh�   AroCP   Arq�P   [�*�Arq��   Arq�P   Art%P   [�=sArtJ�   Art%P   Arv�P   [�X�Arv��   Arv�P   AryP   [ڢAry,�   AryP   Ar{xP   [�n�Ar{��   Ar{xP   Ar}�P   [گ2Ar~�   Ar}�P   Ar�ZP   [ڀMAr��   Ar�ZP   Ar��P   [�D8Ar���   Ar��P   Ar�<P   [�OAr�a�   Ar�<P   Ar��P   [�̛Ar���   Ar��P   Ar�P   [���Ar�C�   Ar�P   Ar��P   [�+�Ar���   Ar��P   Ar� P   [�[�Ar�%�   Ar� P   Ar�qP   [ڟtAr���   Ar�qP   Ar��P   [�bJAr��   Ar��P   Ar�SP   [څ�Ar�x�   Ar�SP   Ar��P   [�p�Ar���   Ar��P   Ar�5P   [�wAr�Z�   Ar�5P   Ar��P   [ڊVAr���   Ar��P   Ar�P   [�W�Ar�<�   Ar�P   Ar��P   [ُ"Ar���   Ar��P   Ar��P   [ٱ+Ar��   Ar��P   Ar�jP   [�A�Ar���   Ar�jP   Ar��P   [��hAr� �   Ar��P   Ar�LP   [��tAr�q�   Ar�LP   Ar��P   [�e{Ar���   Ar��P   Ar�.P   [��sAr�S�   Ar�.P   Ar��P   [��Ar���   Ar��P   Ar�P   [��Ar�5�   Ar�P   Ar��P   [�ǑAr���   Ar��P   Ar��P   [���Ar��   Ar��P   Ar�cP   [�CpAr���   Ar�cP   Ar��P   [��Ar���   Ar��P   Ar�EP   [�UnAr�j�   Ar�EP   ArĶP   [�b�Ar���   ArĶP   Ar�'P   [�aTAr�L�   Ar�'P   ArɘP   [��Arɽ�   ArɘP   Ar�	P   [�1�Ar�.�   Ar�	P   Ar�zP   [�EArΟ�   Ar�zP   Ar��P   [��rAr��   Ar��P   Ar�\P   [�2OArӁ�   Ar�\P   Ar��P   [�$"Ar���   Ar��P   Ar�>P   [�G4Ar�c�   Ar�>P   ArگP   [ج�Ar���   ArگP   Ar� P   [��Ar�E�   Ar� P   ArߑP   [���Ar߶�   ArߑP   Ar�P   [���Ar�'�   Ar�P   Ar�sP   [��Ar��   Ar�sP   Ar��P   [���Ar�	�   Ar��P   Ar�UP   [��fAr�z�   Ar�UP   Ar��P   [�N�Ar���   Ar��P   Ar�7P   [��Ar�\�   Ar�7P   Ar�P   [���Ar���   Ar�P   Ar�P   [ؽjAr�>�   Ar�P   Ar��P   [�`Ar���   Ar��P   Ar��P   [ؖ�Ar� �   Ar��P   Ar�lP   [؄�Ar���   Ar�lP   Ar��P   [؝�Ar��   Ar��P   Ar�NP   [�M�Ar�s�   Ar�NP   As�P   [�<�As��   As�P   As0P   [׏EAsU�   As0P   As�P   [�kAs��   As�P   As	P   [�g�As	7�   As	P   As�P   [�oAs��   As�P   As�P   [�T�As�   As�P   AseP   [ظAs��   AseP   As�P   [�~As��   As�P   AsGP   [�ԧAsl�   AsGP   As�P   [��{As��   As�P   As)P   [�k�AsN�   As)P   As�P   [�uRAs��   As�P   AsP   [ؙVAs0�   AsP   As!|P   [�ԓAs!��   As!|P   As#�P   [ؘ5As$�   As#�P   As&^P   [،!As&��   As&^P   As(�P   [��~As(��   As(�P   As+@P   [���As+e�   As+@P   As-�P   [تKAs-��   As-�P   As0"P   [�\NAs0G�   As0"P   As2�P   [��iAs2��   As2�P   As5P   [ؼtAs5)�   As5P   As7uP   [آKAs7��   As7uP   As9�P   [��aAs:�   As9�P   As<WP   [�$�As<|�   As<WP   As>�P   [؅)As>��   As>�P   AsA9P   [ئ�AsA^�   AsA9P   AsC�P   [�l�AsC��   AsC�P   AsFP   [�q�AsF@�   AsFP   AsH�P   [�16AsH��   AsH�P   AsJ�P   [��AsK"�   AsJ�P   AsMnP   [�@�AsM��   AsMnP   AsO�P   [��_AsP�   AsO�P   AsRPP   [ٓ�AsRu�   AsRPP   AsT�P   [��BAsT��   AsT�P   AsW2P   [�zAsWW�   AsW2P   AsY�P   [�ؑAsY��   AsY�P   As\P   [��fAs\9�   As\P   As^�P   [��4As^��   As^�P   As`�P   [�ܖAsa�   As`�P   AscgP   [�Asc��   AscgP   Ase�P   [�XrAse��   Ase�P   AshIP   [�H�Ashn�   AshIP   Asj�P   [�AYAsj��   Asj�P   Asm+P   [�"jAsmP�   Asm+P   Aso�P   [�K�Aso��   Aso�P   AsrP   [��Asr2�   AsrP   Ast~P   [�tTAst��   Ast~P   Asv�P   [�|%Asw�   Asv�P   Asy`P   [�3�Asy��   Asy`P   As{�P   [�=�As{��   As{�P   As~BP   [�F�As~g�   As~BP   As��P   [�m�As���   As��P   As�$P   [�z|As�I�   As�$P   As��P   [�/OAs���   As��P   As�P   [�]�As�+�   As�P   As�wP   [ڪWAs���   As�wP   As��P   [�nTAs��   As��P   As�YP   [ږHAs�~�   As�YP   As��P   [��As���   As��P   As�;P   [�~�As�`�   As�;P   As��P   [��9As���   As��P   As�P   [���As�B�   As�P   As��P   [��As���   As��P   As��P   [��*As�$�   As��P   As�pP   [ڮvAs���   As�pP   As��P   [ں�As��   As��P   As�RP   [ژ�As�w�   As�RP   As��P   [��WAs���   As��P   As�4P   [ڱ5As�Y�   As�4P   As��P   [ڌ�As���   As��P   As�P   [�r|As�;�   As�P   As��P   [���As���   As��P   As��P   [�]}As��   As��P   As�iP   [��As���   As�iP   As��P   [�tAs���   As��P   As�KP   [�4�As�p�   As�KP   As��P   [� As���   As��P   As�-P   [�
�As�R�   As�-P   AsP   [ڸ�As���   AsP   As�P   [ڣ�As�4�   As�P   AsǀP   [�ǚAsǥ�   AsǀP   As��P   [���As��   As��P   As�bP   [��Aṡ�   As�bP   As��P   [�WAs���   As��P   As�DP   [��As�i�   As�DP   AsӵP   [�'�As���   AsӵP   As�&P   [�B�As�K�   As�&P   AsؗP   [�N�Asؼ�   AsؗP   As�P   [�+HAs�-�   As�P   As�yP   [��Asݞ�   As�yP   As��P   [��As��   As��P   As�[P   [�`4As��   As�[P   As��P   [�2�As���   As��P   As�=P   [�(OAs�b�   As�=P   As�P   [�Q�As���   As�P   As�P   [�h}As�D�   As�P   As�P   [�VAs��   As�P   As�P   [�E�As�&�   As�P   As�rP   [�LAs��   As�rP   As��P   [�/iAs��   As��P   As�TP   [��As�y�   As�TP   As��P   [ۃ"As���   As��P   As�6P   [�LAs�[�   As�6P   As��P   [�M3As���   As��P   AtP   [�M]At=�   AtP   At�P   [��At��   At�P   At�P   [�IAt�   At�P   At	kP   [�c�At	��   At	kP   At�P   [��\At�   At�P   AtMP   [�V�Atr�   AtMP   At�P   [�,�At��   At�P   At/P   [�gAtT�   At/P   At�P   [�$�At��   At�P   AtP   [�?�At6�   AtP   At�P   [��1At��   At�P   At�P   [�1~At�   At�P   AtdP   [��,At��   AtdP   At!�P   [���At!��   At!�P   At$FP   [��At$k�   At$FP   At&�P   [�?�At&��   At&�P   At)(P   [�4!At)M�   At)(P   At+�P   [�At+��   At+�P   At.
P   [�w�At./�   At.
P   At0{P   [�қAt0��   At0{P   At2�P   [�%�At3�   At2�P   At5]P   [ۂ�At5��   At5]P   At7�P   [�At7��   At7�P   At:?P   [�A(At:d�   At:?P   At<�P   [��At<��   At<�P   At?!P   [�@At?F�   At?!P   AtA�P   [�K.AtA��   AtA�P   AtDP   [�< AtD(�   AtDP   AtFtP   [�hAtF��   AtFtP   AtH�P   [�DAtI
�   AtH�P   AtKVP   [��HAtK{�   AtKVP   AtM�P   [�߬AtM��   AtM�P   AtP8P   [��?AtP]�   AtP8P   AtR�P   [���AtR��   AtR�P   AtUP   [�\AtU?�   AtUP   AtW�P   [�5?AtW��   AtW�P   AtY�P   [�AtZ!�   AtY�P   At\mP   [��At\��   At\mP   At^�P   [��At_�   At^�P   AtaOP   [�R�Atat�   AtaOP   Atc�P   [��uAtc��   Atc�P   Atf1P   [���AtfV�   Atf1P   Ath�P   [�I�Ath��   Ath�P   AtkP   [���Atk8�   AtkP   Atm�P   [�-�Atm��   Atm�P   Ato�P   [�,�Atp�   Ato�P   AtrfP   [���Atr��   AtrfP   Att�P   [�.Att��   Att�P   AtwHP   [��Atwm�   AtwHP   Aty�P   [�Aty��   Aty�P   At|*P   [�7�At|O�   At|*P   At~�P   [�'At~��   At~�P   At�P   [���At�1�   At�P   At�}P   [��At���   At�}P   At��P   [ڶCAt��   At��P   At�_P   [ھ�At���   At�_P   At��P   [ڝFAt���   At��P   At�AP   [ڣ_At�f�   At�AP   At��P   [�މAt���   At��P   At�#P   [��dAt�H�   At�#P   At��P   [�ݳAt���   At��P   At�P   [���At�*�   At�P   At�vP   [��At���   At�vP   At��P   [��At��   At��P   At�XP   [�6�At�}�   At�XP   At��P   [�ӾAt���   At��P   At�:P   [�jpAt�_�   At�:P   At��P   [�RAt���   At��P   At�P   [��At�A�   At�P   At��P   [�IAt���   At��P   At��P   [��At�#�   At��P   At�oP   [�	&At���   At�oP   At��P   [���At��   At��P   At�QP   [��^At�v�   At�QP   At��P   [��EAt���   At��P   At�3P   [�ޠAt�X�   At�3P   At��P   [ڭ`At���   At��P   At�P   [��At�:�   At�P   At��P   [ڤ.At���   At��P   At��P   [��8At��   At��P   At�hP   [�V9Atō�   At�hP   At��P   [ژ]At���   At��P   At�JP   [�X�At�o�   At�JP   At̻P   [�wAt���   At̻P   At�,P   [��At�Q�   At�,P   AtѝP   [��At���   AtѝP   At�P   [��LAt�3�   At�P   At�P   [ڥ�At֤�   At�P   At��P   [�At��   At��P   At�aP   [���Atۆ�   At�aP   At��P   [�;�At���   At��P   At�CP   [ڧqAt�h�   At�CP   At�P   [���At���   At�P   At�%P   [ڣ�At�J�   At�%P   At�P   [�SAt��   At�P   At�P   [ڣ�At�,�   At�P   At�xP   [�σAt��   At�xP   At��P   [ڹ�At��   At��P   At�ZP   [��bAt��   At�ZP   At��P   [��At���   At��P   At�<P   [���At�a�   At�<P   At��P   [�+At���   At��P   At�P   [���At�C�   At�P   At��P   [��sAt���   At��P   Au  P   [�իAu %�   Au  P   AuqP   [�CAu��   AuqP   Au�P   [��RAu�   Au�P   AuSP   [�ҁAux�   AuSP   Au	�P   [�ıAu	��   Au	�P   Au5P   [�ʪAuZ�   Au5P   Au�P   [�8Au��   Au�P   AuP   [�~Au<�   AuP   Au�P   [��Au��   Au�P   Au�P   [ڼ�Au�   Au�P   AujP   [�f2Au��   AujP   Au�P   [�*�Au �   Au�P   AuLP   [��>Auq�   AuLP   Au�P   [��Au��   Au�P   Au".P   [�@�Au"S�   Au".P   Au$�P   [�g�Au$��   Au$�P   Au'P   [ڹ�Au'5�   Au'P   Au)�P   [�&Au)��   Au)�P   Au+�P   [��Au,�   Au+�P   Au.cP   [�!�Au.��   Au.cP   Au0�P   [�$�Au0��   Au0�P   Au3EP   [���Au3j�   Au3EP   Au5�P   [��FAu5��   Au5�P   Au8'P   [��Au8L�   Au8'P   Au:�P   [�X�Au:��   Au:�P   Au=	P   [�T�Au=.�   Au=	P   Au?zP   [�!Au?��   Au?zP   AuA�P   [�-�AuB�   AuA�P   AuD\P   [�*AuD��   AuD\P   AuF�P   [�0�AuF��   AuF�P   AuI>P   [���AuIc�   AuI>P   AuK�P   [��AuK��   AuK�P   AuN P   [��bAuNE�   AuN P   AuP�P   [�-AuP��   AuP�P   AuSP   [�$�AuS'�   AuSP   AuUsP   [�k�AuU��   AuUsP   AuW�P   [�@�AuX	�   AuW�P   AuZUP   [��zAuZz�   AuZUP   Au\�P   [�#�Au\��   Au\�P   Au_7P   [�?Au_\�   Au_7P   Aua�P   [�HAua��   Aua�P   AudP   [� MAud>�   AudP   Auf�P   [�KAuf��   Auf�P   Auh�P   [�r�Aui �   Auh�P   AuklP   [�{�Auk��   AuklP   Aum�P   [�u�Aun�   Aum�P   AupNP   [�!�Aups�   AupNP   Aur�P   [�DMAur��   Aur�P   Auu0P   [�N�AuuU�   Auu0P   Auw�P   [�6�Auw��   Auw�P   AuzP   [��Auz7�   AuzP   Au|�P   [�TAu|��   Au|�P   Au~�P   [�Au�   Au~�P   Au�eP   [�?}Au���   Au�eP   Au��P   [�(2Au���   Au��P   Au�GP   [�.�Au�l�   Au�GP   Au��P   [�z�Au���   Au��P   Au�)P   [��pAu�N�   Au�)P   Au��P   [�<}Au���   Au��P   Au�P   [��Au�0�   Au�P   Au�|P   [��Au���   Au�|P   Au��P   [�UAu��   Au��P   Au�^P   [��Au���   Au�^P   Au��P   [�Au���   Au��P   Au�@P   [��Au�e�   Au�@P   Au��P   [�*�Au���   Au��P   Au�"P   [�r�Au�G�   Au�"P   Au��P   [�C"Au���   Au��P   Au�P   [ۨ4Au�)�   Au�P   Au�uP   [�a�Au���   Au�uP   Au��P   [�^�Au��   Au��P   Au�WP   [��Au�|�   Au�WP   Au��P   [�c�Au���   Au��P   Au�9P   [���Au�^�   Au�9P   Au��P   [�KAu���   Au��P   Au�P   [�d�Au�@�   Au�P   Au��P   [�GAu���   Au��P   Au��P   [��Au�"�   Au��P   Au�nP   [��SAu���   Au�nP   Au��P   [ۄ�Au��   Au��P   Au�PP   [��:Au�u�   Au�PP   Au��P   [���Au���   Au��P   Au�2P   [�{	Au�W�   Au�2P   AuʣP   [��Au���   AuʣP   Au�P   [��|Au�9�   Au�P   AuυP   [�}�AuϪ�   AuυP   Au��P   [�P�Au��   Au��P   Au�gP   [�y�AuԌ�   Au�gP   Au��P   [�t�Au���   Au��P   Au�IP   [�/!Au�n�   Au�IP   AuۺP   [�b�Au���   AuۺP   Au�+P   [�!�Au�P�   Au�+P   Au��P   [ۥ�Au���   Au��P   Au�P   [�l�Au�2�   Au�P   Au�~P   [�^>Au��   Au�~P   Au��P   [�5�Au��   Au��P   Au�`P   [۵wAu��   Au�`P   Au��P   [�EcAu���   Au��P   Au�BP   [�W�Au�g�   Au�BP   Au�P   [ۚ