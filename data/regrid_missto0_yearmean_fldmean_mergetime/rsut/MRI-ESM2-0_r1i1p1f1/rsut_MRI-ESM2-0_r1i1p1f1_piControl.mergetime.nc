CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         AJ�       creation_date         2019-02-20T09:29:39Z   data_specs_version        01.00.29   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.piControl.none.r1i1p1f1    grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      �Wed Aug 10 15:19:23 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MRI-ESM2-0_r1i1p1f1/rsut_MRI-ESM2-0_r1i1p1f1_piControl.mergetime.nc
Fri Apr 08 10:34:15 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 10:33:51 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Amon.rsut.gn.v20190222/rsut_Amon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.bic_missto0.yearmean.nc
2019-02-20T09:29:39Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_piControl_100 (sfc_avr_mon.ctl)      initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/5e9f8377-5d08-44b8-be97-c31f45247beb      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   
   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   original_name         USWT   cell_measures         area: areacella    history       r2019-02-20T09:29:39Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).              �                Aq���   Aq��P   Aq�P   B�)Aq�6�   Aq�P   Aq��P   B��@Aq���   Aq��P   Aq��P   B�CAq��   Aq��P   Aq�dP   B��	Aq���   Aq�dP   Aq��P   B�HyAq���   Aq��P   Aq�FP   B���Aq�k�   Aq�FP   Aq��P   B��YAq���   Aq��P   Aq�(P   BóAq�M�   Aq�(P   Aq��P   B�paAq���   Aq��P   Aq�
P   B�PwAq�/�   Aq�
P   Aq�{P   B��Aq���   Aq�{P   Aq��P   B�t<Aq��   Aq��P   Aq�]P   Bþ�AqĂ�   Aq�]P   Aq��P   B��Aq���   Aq��P   Aq�?P   B��BAq�d�   Aq�?P   Aq˰P   B� �Aq���   Aq˰P   Aq�!P   B�BeAq�F�   Aq�!P   AqВP   B�}0Aqз�   AqВP   Aq�P   BÁ�Aq�(�   Aq�P   Aq�tP   B�M�Aqՙ�   Aq�tP   Aq��P   B�DVAq�
�   Aq��P   Aq�VP   B���Aq�{�   Aq�VP   Aq��P   B�F�Aq���   Aq��P   Aq�8P   Bû^Aq�]�   Aq�8P   Aq�P   B�Y�Aq���   Aq�P   Aq�P   B�1.Aq�?�   Aq�P   Aq�P   B�d�Aq��   Aq�P   Aq��P   B�9�Aq�!�   Aq��P   Aq�mP   B��Aq��   Aq�mP   Aq��P   BJAq��   Aq��P   Aq�OP   Bè�Aq�t�   Aq�OP   Aq��P   B�c}Aq���   Aq��P   Aq�1P   B�i�Aq�V�   Aq�1P   Aq��P   B�3YAq���   Aq��P   Aq�P   B�;�Aq�8�   Aq�P   Aq��P   B�6�Aq���   Aq��P   Aq��P   B�v�Aq��   Aq��P   ArfP   BµAr��   ArfP   Ar�P   B½%Ar��   Ar�P   ArHP   B�d�Arm�   ArHP   Ar�P   B��OAr��   Ar�P   Ar*P   B�
�ArO�   Ar*P   Ar�P   B�[�Ar��   Ar�P   ArP   B�s�Ar1�   ArP   Ar}P   B�=�Ar��   Ar}P   Ar�P   B�E�Ar�   Ar�P   Ar_P   B�L+Ar��   Ar_P   Ar�P   B�/Ar��   Ar�P   ArAP   BüBArf�   ArAP   Ar�P   Bà�Ar��   Ar�P   Ar!#P   B��Ar!H�   Ar!#P   Ar#�P   BÑuAr#��   Ar#�P   Ar&P   B��lAr&*�   Ar&P   Ar(vP   B��Ar(��   Ar(vP   Ar*�P   B�|KAr+�   Ar*�P   Ar-XP   B�O5Ar-}�   Ar-XP   Ar/�P   B��1Ar/��   Ar/�P   Ar2:P   BòTAr2_�   Ar2:P   Ar4�P   Bď2Ar4��   Ar4�P   Ar7P   B�T�Ar7A�   Ar7P   Ar9�P   B�'nAr9��   Ar9�P   Ar;�P   BÛ�Ar<#�   Ar;�P   Ar>oP   B�|�Ar>��   Ar>oP   Ar@�P   BùoArA�   Ar@�P   ArCQP   Bà�ArCv�   ArCQP   ArE�P   B�j�ArE��   ArE�P   ArH3P   BîFArHX�   ArH3P   ArJ�P   B�(ArJ��   ArJ�P   ArMP   B��ArM:�   ArMP   ArO�P   BÅ2ArO��   ArO�P   ArQ�P   B�ArR�   ArQ�P   ArThP   B�p9ArT��   ArThP   ArV�P   BêArV��   ArV�P   ArYJP   B���ArYo�   ArYJP   Ar[�P   Bá�Ar[��   Ar[�P   Ar^,P   B��Ar^Q�   Ar^,P   Ar`�P   B�Ar`��   Ar`�P   ArcP   B�i�Arc3�   ArcP   AreP   B�=nAre��   AreP   Arg�P   B�9�Arh�   Arg�P   ArjaP   B��Arj��   ArjaP   Arl�P   B��%Arl��   Arl�P   AroCP   B��Aroh�   AroCP   Arq�P   Bä[Arq��   Arq�P   Art%P   Bñ�ArtJ�   Art%P   Arv�P   BÙ3Arv��   Arv�P   AryP   B�oAry,�   AryP   Ar{xP   B��iAr{��   Ar{xP   Ar}�P   Bă�Ar~�   Ar}�P   Ar�ZP   B�_�Ar��   Ar�ZP   Ar��P   B×�Ar���   Ar��P   Ar�<P   BÊ�Ar�a�   Ar�<P   Ar��P   BÑ�Ar���   Ar��P   Ar�P   B�1�Ar�C�   Ar�P   Ar��P   B�ENAr���   Ar��P   Ar� P   B�b�Ar�%�   Ar� P   Ar�qP   BÅgAr���   Ar�qP   Ar��P   B�8�Ar��   Ar��P   Ar�SP   B��
Ar�x�   Ar�SP   Ar��P   B�u�Ar���   Ar��P   Ar�5P   B�]Ar�Z�   Ar�5P   Ar��P   B��XAr���   Ar��P   Ar�P   B�8�Ar�<�   Ar�P   Ar��P   B�:Ar���   Ar��P   Ar��P   Bë�Ar��   Ar��P   Ar�jP   B�UMAr���   Ar�jP   Ar��P   B��Ar� �   Ar��P   Ar�LP   B���Ar�q�   Ar�LP   Ar��P   B�yAr���   Ar��P   Ar�.P   B��Ar�S�   Ar�.P   Ar��P   B�M�Ar���   Ar��P   Ar�P   B�X�Ar�5�   Ar�P   Ar��P   BÒ�Ar���   Ar��P   Ar��P   BðAr��   Ar��P   Ar�cP   B��Ar���   Ar�cP   Ar��P   B�Ar���   Ar��P   Ar�EP   B��~Ar�j�   Ar�EP   ArĶP   B�}�Ar���   ArĶP   Ar�'P   BïAr�L�   Ar�'P   ArɘP   B��0Arɽ�   ArɘP   Ar�	P   B� �Ar�.�   Ar�	P   Ar�zP   B�H)ArΟ�   Ar�zP   Ar��P   B�+tAr��   Ar��P   Ar�\P   B�#�ArӁ�   Ar�\P   Ar��P   BácAr���   Ar��P   Ar�>P   B�8Ar�c�   Ar�>P   ArگP   B��Ar���   ArگP   Ar� P   BÁ�Ar�E�   Ar� P   ArߑP   B��xAr߶�   ArߑP   Ar�P   B�2�Ar�'�   Ar�P   Ar�sP   B�޾Ar��   Ar�sP   Ar��P   B�#Ar�	�   Ar��P   Ar�UP   B�:sAr�z�   Ar�UP   Ar��P   BøEAr���   Ar��P   Ar�7P   B���Ar�\�   Ar�7P   Ar�P   B�Ar���   Ar�P   Ar�P   B¤Ar�>�   Ar�P   Ar��P   B�͟Ar���   Ar��P   Ar��P   B��EAr� �   Ar��P   Ar�lP   B�Ar���   Ar�lP   Ar��P   BÎ�Ar��   Ar��P   Ar�NP   Bò�Ar�s�   Ar�NP   As�P   BÏAAs��   As�P   As0P   B��AsU�   As0P   As�P   BòAs��   As�P   As	P   B�% As	7�   As	P   As�P   B���As��   As�P   As�P   B��As�   As�P   AseP   BÁEAs��   AseP   As�P   B�oAs��   As�P   AsGP   B��RAsl�   AsGP   As�P   B�T�As��   As�P   As)P   Bæ�AsN�   As)P   As�P   B���As��   As�P   AsP   B���As0�   AsP   As!|P   B�$As!��   As!|P   As#�P   B¨2As$�   As#�P   As&^P   B�RAs&��   As&^P   As(�P   B�wxAs(��   As(�P   As+@P   B�!As+e�   As+@P   As-�P   B��{As-��   As-�P   As0"P   B�$�As0G�   As0"P   As2�P   B�6As2��   As2�P   As5P   BÂUAs5)�   As5P   As7uP   B��As7��   As7uP   As9�P   B�E�As:�   As9�P   As<WP   B��As<|�   As<WP   As>�P   B��As>��   As>�P   AsA9P   B­9AsA^�   AsA9P   AsC�P   B�zkAsC��   AsC�P   AsFP   B�-yAsF@�   AsFP   AsH�P   BÂ�AsH��   AsH�P   AsJ�P   B�"8AsK"�   AsJ�P   AsMnP   Bµ�AsM��   AsMnP   AsO�P   B�d�AsP�   AsO�P   AsRPP   B�@�AsRu�   AsRPP   AsT�P   B�oCAsT��   AsT�P   AsW2P   Bĵ�AsWW�   AsW2P   AsY�P   B�5AsY��   AsY�P   As\P   B�E�As\9�   As\P   As^�P   Bð�As^��   As^�P   As`�P   B��;Asa�   As`�P   AscgP   B�5�Asc��   AscgP   Ase�P   B��Ase��   Ase�P   AshIP   Bÿ�Ashn�   AshIP   Asj�P   B��gAsj��   Asj�P   Asm+P   B��{AsmP�   Asm+P   Aso�P   B�g�Aso��   Aso�P   AsrP   B��JAsr2�   AsrP   Ast~P   B�Ast��   Ast~P   Asv�P   B�uQAsw�   Asv�P   Asy`P   B�?AAsy��   Asy`P   As{�P   B�|�As{��   As{�P   As~BP   B�e�As~g�   As~BP   As��P   B�
As���   As��P   As�$P   B�/�As�I�   As�$P   As��P   B�M`As���   As��P   As�P   B�|As�+�   As�P   As�wP   B¼$As���   As�wP   As��P   B�HUAs��   As��P   As�YP   B�m�As�~�   As�YP   As��P   B��#As���   As��P   As�;P   B��HAs�`�   As�;P   As��P   B�>uAs���   As��P   As�P   B��As�B�   As�P   As��P   B��As���   As��P   As��P   B�9�As�$�   As��P   As�pP   B�VAs���   As�pP   As��P   B�ibAs��   As��P   As�RP   B���As�w�   As�RP   As��P   B�9�As���   As��P   As�4P   B�N4As�Y�   As�4P   As��P   BÕYAs���   As��P   As�P   B�$_As�;�   As�P   As��P   B�1As���   As��P   As��P   BóBAs��   As��P   As�iP   B��tAs���   As�iP   As��P   B�V�As���   As��P   As�KP   BÎAs�p�   As�KP   As��P   B��hAs���   As��P   As�-P   B�{LAs�R�   As�-P   AsP   B�+�As���   AsP   As�P   B���As�4�   As�P   AsǀP   Bæ�Asǥ�   AsǀP   As��P   B�7�As��   As��P   As�bP   B�HAṡ�   As�bP   As��P   Bč�As���   As��P   As�DP   B�As�i�   As�DP   AsӵP   B���As���   AsӵP   As�&P   B���As�K�   As�&P   AsؗP   B�X�Asؼ�   AsؗP   As�P   B×zAs�-�   As�P   As�yP   B�Asݞ�   As�yP   As��P   B��As��   As��P   As�[P   B�4�As��   As�[P   As��P   B�As���   As��P   As�=P   B±bAs�b�   As�=P   As�P   B��2As���   As�P   As�P   B��As�D�   As�P   As�P   B�LuAs��   As�P   As�P   Bµ^As�&�   As�P   As�rP   B��;As��   As�rP   As��P   B��]As��   As��P   As�TP   B�f�As�y�   As�TP   As��P   B��PAs���   As��P   As�6P   B�As�[�   As�6P   As��P   B�v�As���   As��P   AtP   B^At=�   AtP   At�P   B�utAt��   At�P   At�P   Bª�At�   At�P   At	kP   B�%	At	��   At	kP   At�P   B�J^At�   At�P   AtMP   BÍ&Atr�   AtMP   At�P   B��At��   At�P   At/P   B�ʟAtT�   At/P   At�P   B���At��   At�P   AtP   B���At6�   AtP   At�P   BÑzAt��   At�P   At�P   BÅyAt�   At�P   AtdP   B���At��   AtdP   At!�P   Bï�At!��   At!�P   At$FP   Bæ^At$k�   At$FP   At&�P   BÃ�At&��   At&�P   At)(P   B���At)M�   At)(P   At+�P   B��At+��   At+�P   At.
P   B÷AAt./�   At.
P   At0{P   B�d�At0��   At0{P   At2�P   B�m�At3�   At2�P   At5]P   B´�At5��   At5]P   At7�P   B��At7��   At7�P   At:?P   B��cAt:d�   At:?P   At<�P   BAt<��   At<�P   At?!P   BÊ�At?F�   At?!P   AtA�P   B��,AtA��   AtA�P   AtDP   B�.AtD(�   AtDP   AtFtP   B��AtF��   AtFtP   AtH�P   B�1GAtI
�   AtH�P   AtKVP   Bä�AtK{�   AtKVP   AtM�P   B�AtM��   AtM�P   AtP8P   B�LAtP]�   AtP8P   AtR�P   B��AtR��   AtR�P   AtUP   B�w�AtU?�   AtUP   AtW�P   B·tAtW��   AtW�P   AtY�P   B�*lAtZ!�   AtY�P   At\mP   B�!�At\��   At\mP   At^�P   B��At_�   At^�P   AtaOP   B�:rAtat�   AtaOP   Atc�P   B�A�Atc��   Atc�P   Atf1P   B�׭AtfV�   Atf1P   Ath�P   B��4Ath��   Ath�P   AtkP   B�q�Atk8�   AtkP   Atm�P   B�ATAtm��   Atm�P   Ato�P   B��zAtp�   Ato�P   AtrfP   B�� Atr��   AtrfP   Att�P   B¶�Att��   Att�P   AtwHP   B�y�Atwm�   AtwHP   Aty�P   B���Aty��   Aty�P   At|*P   B´�At|O�   At|*P   At~�P   B��At~��   At~�P   At�P   B�7�At�1�   At�P   At�}P   B�ݡAt���   At�}P   At��P   B�r�At��   At��P   At�_P   B�L�At���   At�_P   At��P   BÞ�At���   At��P   At�AP   B��At�f�   At�AP   At��P   B���At���   At��P   At�#P   B�1�At�H�   At�#P   At��P   B�BpAt���   At��P   At�P   B�uAt�*�   At�P   At�vP   B�C�At���   At�vP   At��P   B���At��   At��P   At�XP   B�+�At�}�   At�XP   At��P   Bã�At���   At��P   At�:P   B��At�_�   At�:P   At��P   B�;�At���   At��P   At�P   B�wAt�A�   At�P   At��P   B��At���   At��P   At��P   B��~At�#�   At��P   At�oP   B� �At���   At�oP   At��P   BÀ�At��   At��P   At�QP   B�0�At�v�   At�QP   At��P   BÕiAt���   At��P   At�3P   B��yAt�X�   At�3P   At��P   B�a�At���   At��P   At�P   B�үAt�:�   At�P   At��P   B�At���   At��P   At��P   B��At��   At��P   At�hP   B�$tAtō�   At�hP   At��P   B��At���   At��P   At�JP   B���At�o�   At�JP   At̻P   B�)At���   At̻P   At�,P   B��At�Q�   At�,P   AtѝP   B�F�At���   AtѝP   At�P   BRAt�3�   At�P   At�P   B��HAt֤�   At�P   At��P   B�At��   At��P   At�aP   B� Atۆ�   At�aP   At��P   BÇjAt���   At��P   At�CP   B�vAt�h�   At�CP   At�P   BÝ�At���   At�P   At�%P   B�`(At�J�   At�%P   At�P   B�V�At��   At�P   At�P   B�	At�,�   At�P   At�xP   B���At��   At�xP   At��P   B��NAt��   At��P   At�ZP   B��IAt��   At�ZP   At��P   B�ΏAt���   At��P   At�<P   B�?�At�a�   At�<P   At��P   B�Q�At���   At��P   At�P   B��At�C�   At�P   At��P   B��At���   At��P   Au  P   BÍ�Au %�   Au  P   AuqP   B�\�Au��   AuqP   Au�P   B���Au�   Au�P   AuSP   B���Aux�   AuSP   Au	�P   B¤�Au	��   Au	�P   Au5P   BÐ�AuZ�   Au5P   Au�P   Bó�Au��   Au�P   AuP   B�l�Au<�   AuP   Au�P   B��4Au��   Au�P   Au�P   B�S(Au�   Au�P   AujP   B�'�Au��   AujP   Au�P   B÷+Au �   Au�P   AuLP   BÌ�Auq�   AuLP   Au�P   B��Au��   Au�P   Au".P   B�SoAu"S�   Au".P   Au$�P   B¯�Au$��   Au$�P   Au'P   BÑ�Au'5�   Au'P   Au)�P   B�`Au)��   Au)�P   Au+�P   B�dEAu,�   Au+�P   Au.cP   B�1Au.��   Au.cP   Au0�P   BÑ�Au0��   Au0�P   Au3EP   B��Au3j�   Au3EP   Au5�P   B�4�Au5��   Au5�P   Au8'P   Bô�Au8L�   Au8'P   Au:�P   B�btAu:��   Au:�P   Au=	P   B�>Au=.�   Au=	P   Au?zP   B���Au?��   Au?zP   AuA�P   Bý>AuB�   AuA�P   AuD\P   B�KiAuD��   AuD\P   AuF�P   B��AuF��   AuF�P   AuI>P   B��AuIc�   AuI>P   AuK�P   B�BAuK��   AuK�P   AuN P   B¡�AuNE�   AuN P   AuP�P   B³�AuP��   AuP�P   AuSP   B���AuS'�   AuSP   AuUsP   B��`AuU��   AuUsP   AuW�P   B�K�AuX	�   AuW�P   AuZUP   B��AuZz�   AuZUP   Au\�P   B�hAu\��   Au\�P   Au_7P   BÔ�Au_\�   Au_7P   Aua�P   B�T(Aua��   Aua�P   AudP   B�¸Aud>�   AudP   Auf�P   BÊ�Auf��   Auf�P   Auh�P   B�c�Aui �   Auh�P   AuklP   B�D�Auk��   AuklP   Aum�P   B�Aun�   Aum�P   AupNP   B�ݍAups�   AupNP   Aur�P   B��SAur��   Aur�P   Auu0P   B¨�AuuU�   Auu0P   Auw�P   B��sAuw��   Auw�P   AuzP   B��_Auz7�   AuzP   Au|�P   B��Au|��   Au|�P   Au~�P   B�ZAu�   Au~�P   Au�eP   BðAu���   Au�eP   Au��P   B���Au���   Au��P   Au�GP   B���Au�l�   Au�GP   Au��P   B�VfAu���   Au��P   Au�)P   BÚNAu�N�   Au�)P   Au��P   B�Y�Au���   Au��P   Au�P   B�*�Au�0�   Au�P   Au�|P   B��Au���   Au�|P   Au��P   B�b-Au��   Au��P   Au�^P   B�ZAu���   Au�^P   Au��P   B���Au���   Au��P   Au�@P   B��Au�e�   Au�@P   Au��P   BÂ<Au���   Au��P   Au�"P   B��kAu�G�   Au�"P   Au��P   B�w�Au���   Au��P   Au�P   B�W�Au�)�   Au�P   Au�uP   B�)_Au���   Au�uP   Au��P   B��@Au��   Au��P   Au�WP   B�C�Au�|�   Au�WP   Au��P   B��Au���   Au��P   Au�9P   B�VUAu�^�   Au�9P   Au��P   Bý/Au���   Au��P   Au�P   B>Au�@�   Au�P   Au��P   B�oAu���   Au��P   Au��P   B�FAu�"�   Au��P   Au�nP   B�K�Au���   Au�nP   Au��P   B�d�Au��   Au��P   Au�PP   B���Au�u�   Au�PP   Au��P   B��Au���   Au��P   Au�2P   B��]Au�W�   Au�2P   AuʣP   B���Au���   AuʣP   Au�P   B�x�Au�9�   Au�P   AuυP   B�AuϪ�   AuυP   Au��P   B�$kAu��   Au��P   Au�gP   B�ҒAuԌ�   Au�gP   Au��P   B�^�Au���   Au��P   Au�IP   B�q�Au�n�   Au�IP   AuۺP   B��Au���   AuۺP   Au�+P   B�:UAu�P�   Au�+P   Au��P   B��}Au���   Au��P   Au�P   B��WAu�2�   Au�P   Au�~P   Bó�Au��   Au�~P   Au��P   BúAu��   Au��P   Au�`P   B°�Au��   Au�`P   Au��P   B��uAu���   Au��P   Au�BP   B�s�Au�g�   Au�BP   Au�P   B�͖Au���   Au�P   Au�$P   BîAu�I�   Au�$P   Au��P   B�.�Au���   Au��P   Au�P   Bô�Au�+�   Au�P   Au�wP   B�kAu���   Au�wP   Au��P   B�}�Au��   Au��P   Av YP   B��Av ~�   Av YP   Av�P   B�JQAv��   Av�P   Av;P   B�ݯAv`�   Av;P   Av�P   B�A�Av��   Av�P   Av
P   B�,2Av
B�   Av
P   Av�P   B�b#Av��   Av�P   Av�P   B�M�Av$�   Av�P   AvpP   B�nHAv��   AvpP   Av�P   B�Q�Av�   Av�P   AvRP   B�~�Avw�   AvRP   Av�P   Bå]Av��   Av�P   Av4P   Bè�AvY�   Av4P   Av�P   B��Av��   Av�P   Av P   B��7Av ;�   Av P   Av"�P   B�i|Av"��   Av"�P   Av$�P   B�Av%�   Av$�P   Av'iP   B��Av'��   Av'iP   Av)�P   B�LAv)��   Av)�P   Av,KP   B�7sAv,p�   Av,KP   Av.�P   B�5Av.��   Av.�P   Av1-P   B�XjAv1R�   Av1-P   Av3�P   B�"Av3��   Av3�P   Av6P   BÏ|Av64�   Av6P   Av8�P   B��}Av8��   Av8�P   Av:�P   B�f�Av;�   Av:�P   Av=bP   B�s�Av=��   Av=bP   Av?�P   Bó�Av?��   Av?�P   AvBDP   B��eAvBi�   AvBDP   AvD�P   B�!�AvD��   AvD�P   AvG&P   B�l,AvGK�   AvG&P   AvI�P   B��AvI��   AvI�P   AvLP   B�UcAvL-�   AvLP   AvNyP   BÂ�AvN��   AvNyP   AvP�P   B�	�AvQ�   AvP�P   AvS[P   B�hiAvS��   AvS[P   AvU�P   B/AvU��   AvU�P   AvX=P   B�TtAvXb�   AvX=P   AvZ�P   B�OCAvZ��   AvZ�P   Av]P   B�}Av]D�   Av]P   Av_�P   B�ƻAv_��   Av_�P   AvbP   B�S�Avb&�   AvbP   AvdrP   Bã3Avd��   AvdrP   Avf�P   B��Avg�   Avf�P   AviTP   B�IcAviy�   AviTP   Avk�P   B�=YAvk��   Avk�P   Avn6P   B� �Avn[�   Avn6P   Avp�P   B�8RAvp��   Avp�P   AvsP   B�(�Avs=�   AvsP   Avu�P   B�)nAvu��   Avu�P   Avw�P   B�X�Avx�   Avw�P   AvzkP   B��Avz��   AvzkP   Av|�P   B��hAv}�   Av|�P   AvMP   BþmAvr�   AvMP   Av��P   Bá�Av���   Av��P   Av�/P   B��Av�T�   Av�/P   Av��P   B��7Av���   Av��P   Av�P   B�)Av�6�   Av�P   Av��P   B§�Av���   Av��P   Av��P   B��xAv��   Av��P   Av�dP   B��5Av���   Av�dP   Av��P   B�0Av���   Av��P   Av�FP   B×�Av�k�   Av�FP   Av��P   B�OAv���   Av��P   Av�(P   B�̀Av�M�   Av�(P   Av��P   B�>Av���   Av��P   Av�
P   B�V[Av�/�   Av�
P   Av�{P   B�Av���   Av�{P   Av��P   B�U�Av��   Av��P   Av�]P   B�tFAv���   Av�]P   Av��P   Bí�Av���   Av��P   Av�?P   Bß�Av�d�   Av�?P   Av��P   B¼�Av���   Av��P   Av�!P   B¡�Av�F�   Av�!P   Av��P   B�Av���   Av��P   Av�P   B���Av�(�   Av�P   Av�tP   B�ρAv���   Av�tP   Av��P   B��9Av�
�   Av��P   Av�VP   B�W�Av�{�   Av�VP   Av��P   B�/Av���   Av��P   Av�8P   B�I�Av�]�   Av�8P   AvéP   B���Av���   AvéP   Av�P   B�}�Av�?�   Av�P   AvȋP   B�M+AvȰ�   AvȋP   Av��P   BAv�!�   Av��P   Av�mP   B� pAv͒�   Av�mP   Av��P   B��Av��   Av��P   Av�OP   BöAv�t�   Av�OP   Av��P   B�#Av���   Av��P   Av�1P   B�ZAv�V�   Av�1P   Av٢P   B� �Av���   Av٢P   Av�P   BÊ�Av�8�   Av�P   AvބP   B°�Avީ�   AvބP   Av��P   BÞ�Av��   Av��P   Av�fP   Bþ'Av��   Av�fP   Av��P   B�Q`Av���   Av��P   Av�HP   B��Av�m�   Av�HP   Av�P   Bì�Av���   Av�P   Av�*P   BÛJAv�O�   Av�*P   Av�P   BÈ�Av���   Av�P   Av�P   B�M)Av�1�   Av�P   Av�}P   B��Av���   Av�}P   Av��P   B�/Av��   Av��P   Av�_P   B�Av���   Av�_P   Av��P   BÄAv���   Av��P   Av�AP   Bò�Av�f�   Av�AP   Aw �P   B�:$Aw ��   Aw �P   Aw#P   B��AwH�   Aw#P   Aw�P   B�9Aw��   Aw�P   AwP   B�wjAw*�   AwP   Aw
vP   B�� Aw
��   Aw
vP   Aw�P   B¥�Aw�   Aw�P   AwXP   B��KAw}�   AwXP   Aw�P   B�\	Aw��   Aw�P   Aw:P   B���Aw_�   Aw:P   Aw�P   B��kAw��   Aw�P   AwP   B�ÉAwA�   AwP   Aw�P   B�_WAw��   Aw�P   Aw�P   B�[�Aw#�   Aw�P   Aw oP   B�>�Aw ��   Aw oP   Aw"�P   B�UAw#�   Aw"�P   Aw%QP   Bõ�Aw%v�   Aw%QP   Aw'�P   B� �Aw'��   Aw'�P   Aw*3P   B���Aw*X�   Aw*3P   Aw,�P   B��mAw,��   Aw,�P   Aw/P   Bª~Aw/:�   Aw/P   Aw1�P   B���Aw1��   Aw1�P   Aw3�P   B�|�Aw4�   Aw3�P   Aw6hP   B�ԸAw6��   Aw6hP   Aw8�P   B��Aw8��   Aw8�P   Aw;JP   B�]�Aw;o�   Aw;JP   Aw=�P   B��Aw=��   Aw=�P   Aw@,P   B��zAw@Q�   Aw@,P   AwB�P   B���AwB��   AwB�P   AwEP   B��tAwE3�   AwEP   AwGP   B��AwG��   AwGP   AwI�P   B�=sAwJ�   AwI�P   AwLaP   BaAwL��   AwLaP   AwN�P   B�W AwN��   AwN�P   AwQCP   B²AwQh�   AwQCP   AwS�P   B�u<AwS��   AwS�P   AwV%P   B��IAwVJ�   AwV%P   AwX�P   B§AAwX��   AwX�P   Aw[P   Bï�Aw[,�   Aw[P   Aw]xP   B�Aw]��   Aw]xP   Aw_�P   B�N�Aw`�   Aw_�P   AwbZP   B��Awb�   AwbZP   Awd�P   B¯lAwd��   Awd�P   Awg<P   B�[�Awga�   Awg<P   Awi�P   B�ͶAwi��   Awi�P   AwlP   Bö�AwlC�   AwlP   Awn�P   BîSAwn��   Awn�P   Awq P   B�I�Awq%�   Awq P   AwsqP   B��ZAws��   AwsqP   Awu�P   B�I�Awv�   Awu�P   AwxSP   B�ddAwxx�   AwxSP   Awz�P   Bî�Awz��   Awz�P   Aw}5P   B���Aw}Z�   Aw}5P   Aw�P   B�=Aw��   Aw�P   Aw�P   B�}�Aw�<�   Aw�P   Aw��P   B��Aw���   Aw��P   Aw��P   B­"Aw��   Aw��P   Aw�jP   B�rAw���   Aw�jP   Aw��P   B��LAw� �   Aw��P   Aw�LP   B��Aw�q�   Aw�LP   Aw��P   B�<�Aw���   Aw��P   Aw�.P   B���Aw�S�   Aw�.P   Aw��P   B�!�Aw���   Aw��P   Aw�P   B���Aw�5�   Aw�P   Aw��P   B�WMAw���   Aw��P   Aw��P   B�V�Aw��   Aw��P   Aw�cP   B�1�Aw���   Aw�cP   Aw��P   B��UAw���   Aw��P   Aw�EP   B�Aw�j�   Aw�EP   Aw��P   Bģ�Aw���   Aw��P   Aw�'P   B��Aw�L�   Aw�'P   Aw��P   B��IAw���   Aw��P   Aw�	P   BÊ�Aw�.�   Aw�	P   Aw�zP   B�q�Aw���   Aw�zP   Aw��P   B×dAw��   Aw��P   Aw�\P   B��XAw���   Aw�\P   Aw��P   B�[�Aw���   Aw��P   Aw�>P   BæAw�c�   Aw�>P   Aw��P   BÏ�Aw���   Aw��P   Aw� P   B�xAw�E�   Aw� P   Aw��P   B�./Aw���   Aw��P   Aw�P   B�6Aw�'�   Aw�P   Aw�sP   B�FNAwƘ�   Aw�sP   Aw��P   B���Aw�	�   Aw��P   Aw�UP   B���Aw�z�   Aw�UP   Aw��P   B�OAw���   Aw��P   Aw�7P   B�z�Aw�\�   Aw�7P   AwҨP   B�Aw���   AwҨP   Aw�P   B�v�Aw�>�   Aw�P   Aw׊P   Bê�Awׯ�   Aw׊P   Aw��P   Bà�Aw� �   Aw��P   Aw�lP   B�KCAwܑ�   Aw�lP   Aw��P   B���Aw��   Aw��P   Aw�NP   B�PAw�s�   Aw�NP   Aw�P   B�eAw���   Aw�P   Aw�0P   B�UOAw�U�   Aw�0P   Aw�P   B�z�Aw���   Aw�P   Aw�P   B��\Aw�7�   Aw�P   Aw�P   B¸Aw���   Aw�P   Aw��P   B�"�Aw��   Aw��P   Aw�eP   B�?�Aw��   Aw�eP   Aw��P   B�dZAw���   Aw��P   Aw�GP   B�/LAw�l�   Aw�GP   Aw��P   B bAw���   Aw��P   Aw�)P   B�!Aw�N�   Aw�)P   Aw��P   B�TAw���   Aw��P   AxP   B�&Ax0�   AxP   Ax|P   B×�Ax��   Ax|P   Ax�P   B�>�Ax�   Ax�P   Ax^P   B�|dAx��   Ax^P   Ax
�P   B�g'Ax
��   Ax
�P   Ax@P   Bþ8Axe�   Ax@P   Ax�P   B��_Ax��   Ax�P   Ax"P   B�GcAxG�   Ax"P   Ax�P   B�ʊAx��   Ax�P   AxP   B�8�Ax)�   AxP   AxuP   B²_Ax��   AxuP   Ax�P   B�|Ax�   Ax�P   AxWP   B�g�Ax|�   AxWP   Ax �P   B¨�Ax ��   Ax �P   Ax#9P   BÒ�Ax#^�   Ax#9P   Ax%�P   B�Z�Ax%��   Ax%�P   Ax(P   B�CAx(@�   Ax(P   Ax*�P   B®$Ax*��   Ax*�P   Ax,�P   Bú�Ax-"�   Ax,�P   Ax/nP   B���Ax/��   Ax/nP   Ax1�P   B�EHAx2�   Ax1�P   Ax4PP   B�y�Ax4u�   Ax4PP   Ax6�P   B�)�Ax6��   Ax6�P   Ax92P   B�c�Ax9W�   Ax92P   Ax;�P   B��!Ax;��   Ax;�P   Ax>P   B�N�Ax>9�   Ax>P   Ax@�P   B� �Ax@��   Ax@�P   AxB�P   B�N5AxC�   AxB�P   AxEgP   B�۵AxE��   AxEgP   AxG�P   BÁTAxG��   AxG�P   AxJIP   B�ګAxJn�   AxJIP   AxL�P   Bô�AxL��   AxL�P   AxO+P   BWAxOP�   AxO+P   AxQ�P   B�6AxQ��   AxQ�P   AxTP   BÈ�