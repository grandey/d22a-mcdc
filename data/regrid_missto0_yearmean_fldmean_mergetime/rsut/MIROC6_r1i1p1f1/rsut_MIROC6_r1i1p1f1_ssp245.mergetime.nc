CDF   �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       0MIROC6 (2017): 
aerosol: SPRINTARS6.0
atmos: CCSR AGCM (T85; 256 x 128 longitude/latitude; 81 levels; top level 0.004 hPa)
atmosChem: none
land: MATSIRO6.0
landIce: none
ocean: COCO4.9 (tripolar primarily 1deg; 360 x 256 longitude/latitude; 63 levels; top grid cell 0-2 m)
ocnBgchem: none
seaIce: COCO4.9   institution      QJAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-01-30T03:50:04Z   data_specs_version        01.00.28   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Jhttps://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.historical.none.r1i1p1f1     grid      #native atmosphere T85 Gaussian grid    
grid_label        gn     history      
JWed Aug 10 15:19:16 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MIROC6_r1i1p1f1/rsut_MIROC6_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MIROC6_r1i1p1f1/CMIP6.ScenarioMIP.MIROC.MIROC6.ssp245.r1i1p1f1.Amon.rsut.gn.v20190627/rsut_Amon_MIROC6_ssp245_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MIROC6_r1i1p1f1/rsut_MIROC6_r1i1p1f1_ssp245.mergetime.nc
Wed Aug 10 15:19:15 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_195001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rsut/MIROC6_r1i1p1f1/rsut_MIROC6_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 10:28:46 2022: cdo -O -s -selname,rsut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 10:28:41 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rsut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rsut/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.rsut.gn.v20190311/rsut_Amon_MIROC6_historical_r1i1p1f1_gn_185001-194912.bic_missto0.yearmean.nc
2019-01-30T03:50:04Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.     initialization_index            institution_id        MIROC      mip_era       CMIP6      nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      MIROC6     parent_time_units         days since 3200-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         MIROC6     source_type       	AOGCM AER      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(06 November 2018) MD5:0728c79344e0f262bb76e4f9ff0d9afc      title          MIROC6 output prepared for CMIP6   variable_id       rsut   variant_label         r1i1p1f1   license      !CMIP6 model data produced by MIROC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      cmor_version      3.3.2      tracking_id       1hdl:21.14100/0cb84ec4-6be5-4335-a23d-0ad4711da644      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                      standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   original_name         OSRU   original_units        W/m**2     history       �2019-01-30T03:50:04Z altered by CMOR: Converted units from 'W/m**2' to 'W m-2'. 2019-01-30T03:50:04Z altered by CMOR: replaced missing value flag (-999) with standard missing value (1e+20). 2019-01-30T03:50:04Z altered by CMOR: Inverted axis: lat.    cell_measures         area: areacella                             Aq���   Aq��P   Aq�P   B��Aq�6�   Aq�P   Aq��P   B�!�Aq���   Aq��P   Aq��P   B�!*Aq��   Aq��P   Aq�dP   B�.Aq���   Aq�dP   Aq��P   B؏-Aq���   Aq��P   Aq�FP   BڵnAq�k�   Aq�FP   Aq��P   B�&�Aq���   Aq��P   Aq�(P   B���Aq�M�   Aq�(P   Aq��P   B٢�Aq���   Aq��P   Aq�
P   B���Aq�/�   Aq�
P   Aq�{P   B��IAq���   Aq�{P   Aq��P   B�TAq��   Aq��P   Aq�]P   BڥuAqĂ�   Aq�]P   Aq��P   B���Aq���   Aq��P   Aq�?P   B��oAq�d�   Aq�?P   Aq˰P   B�xAq���   Aq˰P   Aq�!P   B��Aq�F�   Aq�!P   AqВP   B�rMAqз�   AqВP   Aq�P   B�.�Aq�(�   Aq�P   Aq�tP   B��\Aqՙ�   Aq�tP   Aq��P   B�ȔAq�
�   Aq��P   Aq�VP   B���Aq�{�   Aq�VP   Aq��P   Bس�Aq���   Aq��P   Aq�8P   BٻcAq�]�   Aq�8P   Aq�P   B�l�Aq���   Aq�P   Aq�P   B�G�Aq�?�   Aq�P   Aq�P   B�BAq��   Aq�P   Aq��P   B�B�Aq�!�   Aq��P   Aq�mP   B�b�Aq��   Aq�mP   Aq��P   B�<GAq��   Aq��P   Aq�OP   Bه2Aq�t�   Aq�OP   Aq��P   BٍAAq���   Aq��P   Aq�1P   B�a�Aq�V�   Aq�1P   Aq��P   B���Aq���   Aq��P   Aq�P   B�R�Aq�8�   Aq�P   Aq��P   B�`�Aq���   Aq��P   Aq��P   B�C�Aq��   Aq��P   ArfP   B��Ar��   ArfP   Ar�P   Bً�Ar��   Ar�P   ArHP   B�[Arm�   ArHP   Ar�P   B��Ar��   Ar�P   Ar*P   B�D�ArO�   Ar*P   Ar�P   B���Ar��   Ar�P   ArP   B���Ar1�   ArP   Ar}P   B��KAr��   Ar}P   Ar�P   B٥bAr�   Ar�P   Ar_P   BْAr��   Ar_P   Ar�P   Bٽ�Ar��   Ar�P   ArAP   B���Arf�   ArAP   Ar�P   B�b;Ar��   Ar�P   Ar!#P   B�F�Ar!H�   Ar!#P   Ar#�P   B��0Ar#��   Ar#�P   Ar&P   BڥhAr&*�   Ar&P   Ar(vP   B��Ar(��   Ar(vP   Ar*�P   B�XAr+�   Ar*�P   Ar-XP   B�1)Ar-}�   Ar-XP   Ar/�P   B�S�Ar/��   Ar/�P   Ar2:P   B�A�Ar2_�   Ar2:P   Ar4�P   B�i2Ar4��   Ar4�P   Ar7P   B��Ar7A�   Ar7P   Ar9�P   B��aAr9��   Ar9�P   Ar;�P   B�+Ar<#�   Ar;�P   Ar>oP   B��WAr>��   Ar>oP   Ar@�P   B�݁ArA�   Ar@�P   ArCQP   B�? ArCv�   ArCQP   ArE�P   BڗArE��   ArE�P   ArH3P   Bن�ArHX�   ArH3P   ArJ�P   B٨�ArJ��   ArJ�P   ArMP   B��MArM:�   ArMP   ArO�P   B�L�ArO��   ArO�P   ArQ�P   B��4ArR�   ArQ�P   ArThP   Bٛ�ArT��   ArThP   ArV�P   B��ArV��   ArV�P   ArYJP   B�O�ArYo�   ArYJP   Ar[�P   B��[Ar[��   Ar[�P   Ar^,P   B� Ar^Q�   Ar^,P   Ar`�P   B�[�Ar`��   Ar`�P   ArcP   B�HArc3�   ArcP   AreP   B��Are��   AreP   Arg�P   Bډ+Arh�   Arg�P   ArjaP   B� XArj��   ArjaP   Arl�P   B���Arl��   Arl�P   AroCP   B�8Aroh�   AroCP   Arq�P   Bړ�Arq��   Arq�P   Art%P   B�kArtJ�   Art%P   Arv�P   B�}�Arv��   Arv�P   AryP   BَxAry,�   AryP   Ar{xP   BٍHAr{��   Ar{xP   Ar}�P   B��Ar~�   Ar}�P   Ar�ZP   B�*9Ar��   Ar�ZP   Ar��P   B�4)Ar���   Ar��P   Ar�<P   B���Ar�a�   Ar�<P   Ar��P   B��MAr���   Ar��P   Ar�P   BْnAr�C�   Ar�P   Ar��P   B�@)Ar���   Ar��P   Ar� P   B�|�Ar�%�   Ar� P   Ar�qP   B���Ar���   Ar�qP   Ar��P   B�d�Ar��   Ar��P   Ar�SP   B�{Ar�x�   Ar�SP   Ar��P   B�3:Ar���   Ar��P   Ar�5P   B��	Ar�Z�   Ar�5P   Ar��P   B�d�Ar���   Ar��P   Ar�P   Bڕ�Ar�<�   Ar�P   Ar��P   Bڊ�Ar���   Ar��P   Ar��P   B� �Ar��   Ar��P   Ar�jP   B��Ar���   Ar�jP   Ar��P   B��Ar� �   Ar��P   Ar�LP   Bگ�Ar�q�   Ar�LP   Ar��P   BڌjAr���   Ar��P   Ar�.P   B��KAr�S�   Ar�.P   Ar��P   B�sqAr���   Ar��P   Ar�P   B�2�Ar�5�   Ar�P   Ar��P   B��Ar���   Ar��P   Ar��P   B�7EAr��   Ar��P   Ar�cP   B��0Ar���   Ar�cP   Ar��P   BܾAr���   Ar��P   Ar�EP   B��CAr�j�   Ar�EP   ArĶP   B�/Ar���   ArĶP   Ar�'P   B۬�Ar�L�   Ar�'P   ArɘP   B��Arɽ�   ArɘP   Ar�	P   B�6uAr�.�   Ar�	P   Ar�zP   B�t�ArΟ�   Ar�zP   Ar��P   B�0_Ar��   Ar��P   Ar�\P   Bڐ.ArӁ�   Ar�\P   Ar��P   B�m�Ar���   Ar��P   Ar�>P   Bܓ&Ar�c�   Ar�>P   ArگP   B�Q�Ar���   ArگP   Ar� P   B��Ar�E�   Ar� P   ArߑP   Bۭ�Ar߶�   ArߑP   Ar�P   B��=Ar�'�   Ar�P   Ar�sP   B�%Ar��   Ar�sP   Ar��P   B�YAr�	�   Ar��P   Ar�UP   B�vAr�z�   Ar�UP   Ar��P   Bݦ1Ar���   Ar��P   Ar�7P   B�`�Ar�\�   Ar�7P   Ar�P   B��LAr���   Ar�P   Ar�P   B�`�Ar�>�   Ar�P   Ar��P   B�_;Ar���   Ar��P   Ar��P   B��gAr� �   Ar��P   Ar�lP   B۝�Ar���   Ar�lP   Ar��P   B�+�Ar��   Ar��P   Ar�NP   B�9�Ar�s�   Ar�NP   As�P   B��As��   As�P   As0P   B�%&AsU�   As0P   As�P   B���As��   As�P   As	P   B�s2As	7�   As	P   As�P   BۡAs��   As�P   As�P   Bۙ�As�   As�P   AseP   B� �As��   AseP   As�P   B�]As��   As�P   AsGP   B�X�Asl�   AsGP   As�P   B��As��   As�P   As)P   Bۑ�AsN�   As)P   As�P   B�kSAs��   As�P   AsP   B۰RAs0�   AsP   As!|P   Bۤ=As!��   As!|P   As#�P   Bۇ�As$�   As#�P   As&^P   B�J�As&��   As&^P   As(�P   Bڕ�As(��   As(�P   As+@P   B�}As+e�   As+@P   As-�P   B�O�As-��   As-�P   As0"P   B�0�As0G�   As0"P   As2�P   B���As2��   As2�P   As5P   B��As5)�   As5P   As7uP   B�qyAs7��   As7uP   As9�P   B�"�As:�   As9�P   As<WP   Bڴ2As<|�   As<WP   As>�P   BڲAs>��   As>�P   AsA9P   B�9AsA^�   AsA9P   AsC�P   BڛgAsC��   AsC�P   AsFP   B��lAsF@�   AsFP   AsH�P   B��,AsH��   AsH�P   AsJ�P   B�AsK"�   AsJ�P   AsMnP   B�{;AsM��   AsMnP   AsO�P   B�ǕAsP�   AsO�P   AsRPP   BټUAsRu�   AsRPP   AsT�P   B�AsT��   AsT�P   AsW2P   Bً�AsWW�   AsW2P   AsY�P   B�,;AsY��   AsY�P   As\P   B�+�As\9�   As\P   As^�P   B�B�As^��   As^�P   As`�P   BًPAsa�   As`�P   AscgP   Bإ�Asc��   AscgP   Ase�P   B�f�Ase��   Ase�P   AshIP   B�0PAshn�   AshIP   Asj�P   Bپ8Asj��   Asj�P   Asm+P   BضAsmP�   Asm+P   Aso�P   B���Aso��   Aso�P   AsrP   B��=Asr2�   AsrP   Ast~P   B�<hAst��   Ast~P   Asv�P   B�M�Asw�   Asv�P   Asy`P   B��Asy��   Asy`P   As{�P   B��As{��   As{�P   As~BP   B�	>As~g�   As~BP   As��P   B�`kAs���   As��P   As�$P   Bر�As�I�   As�$P   As��P   B�14As���   As��P   As�P   B��4As�+�   As�P   As�wP   B��As���   As�wP   As��P   B�wzAs��   As��P   As�YP   B�z�As�~�   As�YP   As��P   B��As���   As��P   As�;P   BײZAs�`�   As�;P   As��P   B�etAs���   As��P   As�P   B�mAs�B�   As�P   As��P   B�:hAs���   As��P   As��P   B�_As�$�   As��P   As�pP   B�d)As���   As�pP   As��P   B��As��   As��P   As�RP   B�>As�w�   As�RP   As��P   B׉�As���   As��P   As�4P   B׏4As�Y�   As�4P   As��P   B�`�As���   As��P   As�P   B��iAs�;�   As�P   As��P   B�NAs���   As��P   As��P   B֯As��   As��P   As�iP   Bף�As���   As�iP   As��P   B���As���   As��P   As�KP   Bֻ�As�p�   As�KP   As��P   B�g�As���   As��P   As�-P   Bֳ�As�R�   As�-P   AsP   B� NAs���   AsP   As�P   B�]As�4�   As�P   AsǀP   B��Asǥ�   AsǀP   As��P   B��As��   As��P   As�bP   Bִ@Aṡ�   As�bP   As��P   Bֶ�As���   As��P   As�DP   BֺAs�i�   As�DP   AsӵP   B֢�As���   AsӵP   As�&P   B֛rAs�K�   As�&P   AsؗP   B��QAsؼ�   AsؗP   As�P   B�v�As�-�   As�P   As�yP   B�%�Asݞ�   As�yP   As��P   B��As��   As��P   As�[P   B��As��   As�[P   As��P   B�-�As���   As��P   As�=P   B�
�As�b�   As�=P   As�P   B���As���   As�P   As�P   B��yAs�D�   As�P   As�P   B�S�As��   As�P   As�P   B���As�&�   As�P   As�rP   B�-LAs��   As�rP   As��P   B�y�As��   As��P   As�TP   B��As�y�   As�TP   As��P   B�ViAs���   As��P   As�6P   B�*�As�[�   As�6P   As��P   B��As���   As��P   AtP   B�F�At=�   AtP   At�P   B��/At��   At�P   At�P   B�WAt�   At�P   At	kP   B֋/