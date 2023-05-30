CDF  �   
      time       bnds      lon       lat          7   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       CNRM-ESM2-1 (2017):  aerosol: TACTIC_v2 atmos: Arpege 6.3 (T127; Gaussian Reduced with 24572 grid points in total distributed over 128 latitude circles (with 256 grid points per latitude circle between 30degN and 30degS reducing to 20 grid points per latitude circle at 88.9degN and 88.9degS); 91 levels; top level 78.4 km) atmosChem: REPROBUS-C_v2 land: Surfex 8.0c ocean: Nemo 3.6 (eORCA1, tripolar primarily 1deg; 362 x 294 longitude/latitude; 75 levels; top grid cell 0-1 m) ocnBgchem: Pisces 2.s seaIce: Gelato 6.1    institution       �CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (Centre Europeen de Recherche et de Formation Avancee en Calcul Scientifique, Toulouse 31057, France)    creation_date         2018-04-23T09:02:20Z   description       DECK: control      title         <CNRM-ESM2-1 model output prepared for CMIP6 / CMIP piControl   activity_id       CMIP   contact       contact.cmip@meteo.fr      data_specs_version        01.00.21   dr2xml_version        1.1    experiment_id         	piControl      
experiment        pre-industrial control     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CNRM-CERFACS.CNRM-ESM2-1.piControl.none.r1i1p1f2      grid      2native ocean tri-polar grid with 105 k ocean cells     
grid_label        gn     nominal_resolution        100 km     initialization_index            institution_id        CNRM-CERFACS   license      ZCMIP6 model data produced by CNRM-CERFACS is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at http://www.umr-cnrm.fr/cmip6/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.     mip_era       CMIP6      parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_activity_id        CMIP   parent_source_id      CNRM-ESM2-1    parent_time_units         days since 1850-01-01 00:00:00     parent_variant_label      r1i1p1f2   branch_method         standard   branch_time_in_parent         @�Հ       branch_time_in_child                 physics_index               product       model-output   realization_index               realm         ocean      
references        'http://www.umr-cnrm.fr/cmip6/references    	source_id         CNRM-ESM2-1    source_type       AOGCM BGC AER CHEM     sub_experiment_id         none   sub_experiment        none   table_id      Omon   variable_id       hfds   variant_info      �. Information provided by this attribute may in some cases be flawed. Users can find more comprehensive and up-to-date documentation via the further_info_url global attribute.    variant_label         r1i1p1f2   EXPID         CNRM-ESM2-1_piControl_r1i1p1f2     CMIP6_CV_version      cv=6.2.3.0-7-g2019642      dr2xml_md5sum          87374385b726e2a5f1e17b33af88ce8c   xios_commit       1442-shuffle   nemo_gelato_commit        49095b3accd5d4c_6524fe19b00467a    arpege_minor_version      6.3.1      history      �Tue May 30 16:59:07 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.yearmean.mul.areacello_piControl_v20181115.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/CNRM-ESM2-1_r1i1p1f2/hfds_CNRM-ESM2-1_r1i1p1f2_piControl.mergetime.nc
Thu Nov 03 22:36:22 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.nc /Users/benjamin/Data/p22b/CMIP6/areacello/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Ofx.areacello.gn.v20181115/areacello_Ofx_CNRM-ESM2-1_piControl_r1i1p1f2_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/CNRM-ESM2-1_r1i1p1f2/CMIP6.CMIP.CNRM-CERFACS.CNRM-ESM2-1.piControl.r1i1p1f2.Omon.hfds.gn.v20181115/hfds_Omon_CNRM-ESM2-1_piControl_r1i1p1f2_gn_185001-234912.yearmean.mul.areacello_piControl_v20181115.fldsum.nc
none    tracking_id       1hdl:21.14100/a6aed716-b08f-404a-8c93-21f5b0f32abf      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         	Time axis      bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               d   	time_bnds                                 l   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               T   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               \   hfds                      standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    online_operation      average    interval_operation        1800 s     interval_write        1 month    description       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any "flux adjustment") .    positive      down   history       none   cell_measures         area: areacello             |eacello             |                Aq���   Aq��P   Aq�P   �ɹAq�6�   Aq�P   Aq��P   WR5Aq���   Aq��P   Aq��P   �'g�Aq��   Aq��P   Aq�dP   W��RAq���   Aq�dP   Aq��P   �
~�Aq���   Aq��P   Aq�FP   �l}�Aq�k�   Aq�FP   Aq��P   V��Aq���   Aq��P   Aq�(P   �٣EAq�M�   Aq�(P   Aq��P   ��!�Aq���   Aq��P   Aq�
P   �M�Aq�/�   Aq�
P   Aq�{P   W��Aq���   Aq�{P   Aq��P   �[ԒAq��   Aq��P   Aq�]P   V�lAAqĂ�   Aq�]P   Aq��P   �i�,Aq���   Aq��P   Aq�?P   ։�%Aq�d�   Aq�?P   Aq˰P   U�VLAq���   Aq˰P   Aq�!P   V��DAq�F�   Aq�!P   AqВP   T���Aqз�   AqВP   Aq�P   UD�pAq�(�   Aq�P   Aq�tP   ֑�Aqՙ�   Aq�tP   Aq��P   Wa|�Aq�
�   Aq��P   Aq�VP   ն8�Aq�{�   Aq�VP   Aq��P   ��3Aq���   Aq��P   Aq�8P   ӵ��Aq�]�   Aq�8P   Aq�P   V�P\Aq���   Aq�P   Aq�P   �7�YAq�?�   Aq�P   Aq�P   W�}Aq��   Aq�P   Aq��P   W~�#Aq�!�   Aq��P   Aq�mP   Ս۔Aq��   Aq�mP   Aq��P   ��AAq��   Aq��P   Aq�OP   V�L�Aq�t�   Aq�OP   Aq��P   V7�bAq���   Aq��P   Aq�1P   V��>Aq�V�   Aq�1P   Aq��P   WJ��Aq���   Aq��P   Aq�P   V��}Aq�8�   Aq�P   Aq��P   U��Aq���   Aq��P   Aq��P   V���Aq��   Aq��P   ArfP   ��r�Ar��   ArfP   Ar�P   V��vAr��   Ar�P   ArHP   ֫�jArm�   ArHP   Ar�P   ����Ar��   Ar�P   Ar*P   U7��ArO�   Ar*P   Ar�P   �ON�Ar��   Ar�P   ArP   ���Ar1�   ArP   Ar}P   V�UAr��   Ar}P   Ar�P   �B"�Ar�   Ar�P   Ar_P   WpoAr��   Ar_P   Ar�P   �H(�Ar��   Ar�P   ArAP   �MR!Arf�   ArAP   Ar�P   ֑
dAr��   Ar�P   Ar!#P   W_�Ar!H�   Ar!#P   Ar#�P   �\�dAr#��   Ar#�P   Ar&P   VJ��Ar&*�   Ar&P   Ar(vP   �8�zAr(��   Ar(vP   Ar*�P   V��sAr+�   Ar*�P   Ar-XP   ֓��Ar-}�   Ar-XP   Ar/�P   Vp6Ar/��   Ar/�P   Ar2:P   VF��Ar2_�   Ar2:P   Ar4�P   W~�^Ar4��   Ar4�P   Ar7P   V;��Ar7A�   Ar7P   Ar9�P   �w4�Ar9��   Ar9�P   Ar;�P   ��4zAr<#�   Ar;�P   Ar>oP   V���Ar>��   Ar>oP   Ar@�P   V�U`ArA�   Ar@�P   ArCQP   V'DArCv�   ArCQP   ArE�P   ֠��ArE��   ArE�P   ArH3P   V� ArHX�   ArH3P   ArJ�P   W"+�ArJ��   ArJ�P   ArMP   ֱ��ArM:�   ArMP   ArO�P   W<��ArO��   ArO�P   ArQ�P   V� ArR�   ArQ�P   ArThP   ���OArT��   ArThP   ArV�P   W8qoArV��   ArV�P   ArYJP   V�5+ArYo�   ArYJP   Ar[�P   V)�'Ar[��   Ar[�P   Ar^,P   ��E�Ar^Q�   Ar^,P   Ar`�P   ֜�Ar`��   Ar`�P   ArcP   �٨�Arc3�   ArcP   AreP   U�1�Are��   AreP   Arg�P   UgQ�Arh�   Arg�P   ArjaP   W�qArj��   ArjaP   Arl�P   VY�Arl��   Arl�P   AroCP   T�ҳAroh�   AroCP   Arq�P   ց��Arq��   Arq�P   Art%P   WG�ArtJ�   Art%P   Arv�P   �0 �Arv��   Arv�P   AryP   V��dAry,�   AryP   Ar{xP   V��!Ar{��   Ar{xP   Ar}�P   W�#Ar~�   Ar}�P   Ar�ZP   VQk_Ar��   Ar�ZP   Ar��P   ׅ��Ar���   Ar��P   Ar�<P   Wɽ�Ar�a�   Ar�<P   Ar��P   V5�vAr���   Ar��P   Ar�P   ֥fAr�C�   Ar�P   Ar��P   W�0Ar���   Ar��P   Ar� P   ն�Ar�%�   Ar� P   Ar�qP   VB��Ar���   Ar�qP   Ar��P   ��]�Ar��   Ar��P   Ar�SP   W+�Ar�x�   Ar�SP   Ar��P   ��<�Ar���   Ar��P   Ar�5P   VH�qAr�Z�   Ar�5P   Ar��P   �/�Ar���   Ar��P   Ar�P   U���Ar�<�   Ar�P   Ar��P   T�A�Ar���   Ar��P   Ar��P   VjAr��   Ar��P   Ar�jP   V�=Ar���   Ar�jP   Ar��P   �9ٯAr� �   Ar��P   Ar�LP   ��Ar�q�   Ar�LP   Ar��P   V�sAr���   Ar��P   Ar�.P   Wz�Ar�S�   Ar�.P   Ar��P   �o�IAr���   Ar��P   Ar�P   V���Ar�5�   Ar�P   Ar��P   V���Ar���   Ar��P   Ar��P   V��WAr��   Ar��P   Ar�cP   �{�Ar���   Ar�cP   Ar��P   V� �Ar���   Ar��P   Ar�EP   W��Ar�j�   Ar�EP   ArĶP   ��0Ar���   ArĶP   Ar�'P   �<R`Ar�L�   Ar�'P   ArɘP   V�JpArɽ�   ArɘP   Ar�	P   V��GAr�.�   Ar�	P   Ar�zP   �1ArΟ�   Ar�zP   Ar��P   �.*KAr��   Ar��P   Ar�\P   ֝�JArӁ�   Ar�\P   Ar��P   W��Ar���   Ar��P   Ar�>P   UB��Ar�c�   Ar�>P   ArگP   �pykAr���   ArگP   Ar� P   W4Ar�E�   Ar� P   ArߑP   V�Ar߶�   ArߑP   Ar�P   V���Ar�'�   Ar�P   Ar�sP   TĀ�Ar��   Ar�sP   Ar��P   ���5Ar�	�   Ar��P   Ar�UP   ִ [Ar�z�   Ar�UP   Ar��P   WVPAr���   Ar��P   Ar�7P   WoAr�\�   Ar�7P   Ar�P   V���Ar���   Ar�P   Ar�P   � ��Ar�>�   Ar�P   Ar��P   W'1GAr���   Ar��P   Ar��P   V���Ar� �   Ar��P   Ar�lP   ��{Ar���   Ar�lP   Ar��P   �u=UAr��   Ar��P   Ar�NP   U�8�Ar�s�   Ar�NP   As�P   W(��As��   As�P   As0P   V�L�AsU�   As0P   As�P   V�6As��   As�P   As	P   WN8As	7�   As	P   As�P   ��As��   As�P   As�P   �\,As�   As�P   AseP   V�
IAs��   AseP   As�P   Wm�aAs��   As�P   AsGP   W![�Asl�   AsGP   As�P   U|�As��   As�P   As)P   �ÌAsN�   As)P   As�P   ֏��As��   As�P   AsP   W?�As0�   AsP   As!|P   �UAAs!��   As!|P   As#�P   �2lAs$�   As#�P   As&^P   �GF9As&��   As&^P   As(�P   U��As(��   As(�P   As+@P   �!eAs+e�   As+@P   As-�P   V��As-��   As-�P   As0"P   �B��As0G�   As0"P   As2�P   �Y��As2��   As2�P   As5P   W��.As5)�   As5P   As7uP   �.¦As7��   As7uP   As9�P   �M�As:�   As9�P   As<WP   �U3�As<|�   As<WP   As>�P   V�L�As>��   As>�P   AsA9P   �Y �AsA^�   AsA9P   AsC�P   U��SAsC��   AsC�P   AsFP   ��
2AsF@�   AsFP   AsH�P   V�Q�AsH��   AsH�P   AsJ�P   ֘�FAsK"�   AsJ�P   AsMnP   V�AsM��   AsMnP   AsO�P   �BƸAsP�   AsO�P   AsRPP   V�FAAsRu�   AsRPP   AsT�P   Wj�AsT��   AsT�P   AsW2P   �fc�AsWW�   AsW2P   AsY�P   U�.�AsY��   AsY�P   As\P   V�As\9�   As\P   As^�P   ԭ1}As^��   As^�P   As`�P   V��Asa�   As`�P   AscgP   �R*]Asc��   AscgP   Ase�P   U⁳Ase��   Ase�P   AshIP   Vz4�Ashn�   AshIP   Asj�P   V��rAsj��   Asj�P   Asm+P   W Y6AsmP�   Asm+P   Aso�P   �AU	Aso��   Aso�P   AsrP   V��^Asr2�   AsrP   Ast~P   ��;oAst��   Ast~P   Asv�P   �*/Asw�   Asv�P   Asy`P   WxJAsy��   Asy`P   As{�P   �	^�As{��   As{�P   As~BP   UA�\As~g�   As~BP   As��P   Tܩ�As���   As��P   As�$P   V��YAs�I�   As�$P   As��P   ��k)As���   As��P   As�P   W2 ZAs�+�   As�P   As�wP   �2�?As���   As�wP   As��P   V��/As��   As��P   As�YP   VީtAs�~�   As�YP   As��P   ֋�cAs���   As��P   As�;P   ԧs�As�`�   As�;P   As��P   ��&As���   As��P   As�P   �"As�B�   As�P   As��P   W��As���   As��P   As��P   TL��As�$�   As��P   As�pP   ��EAs���   As�pP   As��P   V��As��   As��P   As�RP   �MR�As�w�   As�RP   As��P   ��As���   As��P   As�4P   V�$OAs�Y�   As�4P   As��P   �C#kAs���   As��P   As�P   VK�As�;�   As�P   As��P   V�JAs���   As��P   As��P   Vw�mAs��   As��P   As�iP   փ�As���   As�iP   As��P   V���As���   As��P   As�KP   W�As�p�   As�KP   As��P   �]DhAs���   As��P   As�-P   V���As�R�   As�-P   AsP   �}�#As���   AsP   As�P   �}8,As�4�   As�P   AsǀP   V���Asǥ�   AsǀP   As��P   ��-�As��   As��P   As�bP   V=U(Aṡ�   As�bP   As��P   U�x�As���   As��P   As�DP   ֥As�i�   As�DP   AsӵP   �g�hAs���   AsӵP   As�&P   V� �As�K�   As�&P   AsؗP   ���Asؼ�   AsؗP   As�P   V��As�-�   As�P   As�yP   �7�Asݞ�   As�yP   As��P   �<~As��   As��P   As�[P   ���As��   As�[P   As��P   W^AAs���   As��P   As�=P   ���*As�b�   As�=P   As�P   Ty��As���   As�P   As�P   �ndAs�D�   As�P   As�P   ��V�As��   As�P   As�P   Wb"2As�&�   As�P   As�rP   To�(As��   As�rP   As��P   ���)As��   As��P   As�TP   WW
�As�y�   As�TP   As��P   V�-oAs���   As��P   As�6P   ֡�]As�[�   As�6P   As��P   V�9�As���   As��P   AtP   WT�3At=�   AtP   At�P   ��v�At��   At�P   At�P   U
��At�   At�P   At	kP   W8�At	��   At	kP   At�P   �s�At�   At�P   AtMP   ���Atr�   AtMP   At�P   �1�At��   At�P   At/P   V���AtT�   At/P   At�P   WG�At��   At�P   AtP   W@��At6�   AtP   At�P   V���At��   At�P   At�P   ֎0At�   At�P   AtdP   Wr�At��   AtdP   At!�P   ���At!��   At!�P   At$FP   U�-(At$k�   At$FP   At&�P   T�*�At&��   At&�P   At)(P   W��At)M�   At)(P   At+�P   W0�}At+��   At+�P   At.
P   ֖I�At./�   At.
P   At0{P   S��At0��   At0{P   At2�P   W�T�At3�   At2�P   At5]P   ��'tAt5��   At5]P   At7�P   WBaoAt7��   At7�P   At:?P   U���At:d�   At:?P   At<�P   W'-�At<��   At<�P   At?!P   V���At?F�   At?!P   AtA�P   V�T8AtA��   AtA�P   AtDP   �w��AtD(�   AtDP   AtFtP   W=aAtF��   AtFtP   AtH�P   �[�AtI
�   AtH�P   AtKVP   WSfAtK{�   AtKVP   AtM�P   W18�AtM��   AtM�P   AtP8P   ָ��AtP]�   AtP8P   AtR�P   WdY�AtR��   AtR�P   AtUP   ՜ъAtU?�   AtUP   AtW�P   ֘��AtW��   AtW�P   AtY�P   W+ˑAtZ!�   AtY�P   At\mP   �-�At\��   At\mP   At^�P   W,>�At_�   At^�P   AtaOP   V�oAtat�   AtaOP   Atc�P   �� �Atc��   Atc�P   Atf1P   V�k;AtfV�   Atf1P   Ath�P   Vj�lAth��   Ath�P   AtkP   V���Atk8�   AtkP   Atm�P   U�{�Atm��   Atm�P   Ato�P   V�C�Atp�   Ato�P   AtrfP   W�Atr��   AtrfP   Att�P   U��rAtt��   Att�P   AtwHP   W^^�Atwm�   AtwHP   Aty�P   V-?Aty��   Aty�P   At|*P   W�At|O�   At|*P   At~�P   T���At~��   At~�P   At�P   W``]At�1�   At�P   At�}P   ��NAt���   At�}P   At��P   V&�At��   At��P   At�_P   UA,At���   At�_P   At��P   V�_�At���   At��P   At�AP   Wb@�At�f�   At�AP   At��P   ���At���   At��P   At�#P   �*VAt�H�   At�#P   At��P   WY��At���   At��P   At�P   W�At�*�   At�P   At�vP   ֯P�At���   At�vP   At��P   V~�At��   At��P   At�XP   V�At�}�   At�XP   At��P   T,<At���   At��P   At�:P   W(J�At�_�   At�:P   At��P   ֤
�At���   At��P   At�P   U	E$At�A�   At�P   At��P   ��TAt���   At��P   At��P   V9�At�#�   At��P   At�oP   V��At���   At�oP   At��P   ��hYAt��   At��P   At�QP   U�M�At�v�   At�QP   At��P   W1n�At���   At��P   At�3P   ����At�X�   At�3P   At��P   U^�At���   At��P   At�P   W%l�At�:�   At�P   At��P   U��At���   At��P   At��P   W�pAt��   At��P   At�hP   կ��Atō�   At�hP   At��P   W9�At���   At��P   At�JP   ք�	At�o�   At�JP   At̻P   VbY�At���   At̻P   At�,P   V�&�At�Q�   At�,P   AtѝP   ֋U�At���   AtѝP   At�P   V���At�3�   At�P   At�P   Vg��At֤�   At�P   At��P   �QK�At��   At��P   At�aP   W\�UAtۆ�   At�aP   At��P   W1�fAt���   At��P   At�CP   ֑~At�h�   At�CP   At�P   WCY�At���   At�P   At�%P   W8�9At�J�   At�%P   At�P   �7KeAt��   At�P   At�P   V��At�,�   At�P   At�xP   Ո�fAt��   At�xP   At��P   U���At��   At��P   At�ZP   �	At��   At�ZP   At��P   Ք=�At���   At��P   At�<P   WFAt�a�   At�<P   At��P   �At���   At��P   At�P   V~CAt�C�   At�P   At��P   WLAt���   At��P   Au  P   �e�|Au %�   Au  P   AuqP   ֙��Au��   AuqP   Au�P   օ��Au�   Au�P   AuSP   W�bwAux�   AuSP   Au	�P   �f&3Au	��   Au	�P   Au5P   �9M�AuZ�   Au5P   Au�P   ֿX�Au��   Au�P   AuP   V~�Au<�   AuP   Au�P   U"�Au��   Au�P   Au�P   VJ^(Au�   Au�P   AujP   U�xXAu��   AujP   Au�P   Td�Au �   Au�P   AuLP   V�?Auq�   AuLP   Au�P   � �Au��   Au�P   Au".P   UT:�Au"S�   Au".P   Au$�P   Ve�Au$��   Au$�P   Au'P   ��9�Au'5�   Au'P   Au)�P   ��=cAu)��   Au)�P   Au+�P   ����Au,�   Au+�P   Au.cP   ֎�Au.��   Au.cP   Au0�P   V�UAu0��   Au0�P   Au3EP   V���Au3j�   Au3EP   Au5�P   V93Au5��   Au5�P   Au8'P   ֱ��Au8L�   Au8'P   Au:�P   ���Au:��   Au:�P   Au=	P   V�'�Au=.�   Au=	P   Au?zP   U�1�Au?��   Au?zP   AuA�P   ���4AuB�   AuA�P   AuD\P   ֙7{AuD��   AuD\P   AuF�P   �)��AuF��   AuF�P   AuI>P   V�ogAuIc�   AuI>P   AuK�P   W#)�AuK��   AuK�P   AuN P   ��jAuNE�   AuN P   AuP�P   ֽW�AuP��   AuP�P   AuSP   T�pwAuS'�   AuSP   AuUsP   U�K�AuU��   AuUsP   AuW�P   VǣAuX	�   AuW�P   AuZUP   �2�AuZz�   AuZUP   Au\�P   V[�PAu\��   Au\�P   Au_7P   ��סAu_\�   Au_7P   Aua�P   V#/7Aua��   Aua�P   AudP   U�/Aud>�   AudP   Auf�P   VD`�Auf��   Auf�P   Auh�P   �h��Aui �   Auh�P   AuklP   V�,�Auk��   AuklP   Aum�P   V�2RAun�   Aum�P   AupNP   �6�Aups�   AupNP   Aur�P   W|Aur��   Aur�P   Auu0P   ��)�AuuU�   Auu0P   Auw�P   V�4Auw��   Auw�P   AuzP   ��JAuz7�   AuzP   Au|�P   Wh~Au|��   Au|�P   Au~�P   U..nAu�   Au~�P   Au�eP   W<Au���   Au�eP   Au��P   � �Au���   Au��P   Au�GP   �ZN�Au�l�   Au�GP   Au��P   W5�Au���   Au��P   Au�)P   V�cAu�N�   Au�)P   Au��P   ��fSAu���   Au��P   Au�P   �'�Au�0�   Au�P   Au�|P   W&J:Au���   Au�|P   Au��P   �a�iAu��   Au��P   Au�^P   ׎+Au���   Au�^P   Au��P   V(K�Au���   Au��P   Au�@P   Ww�_Au�e�   Au�@P   Au��P   ֻ�Au���   Au��P   Au�"P   Vy�Au�G�   Au�"P   Au��P   ��BoAu���   Au��P   Au�P   V"��Au�)�   Au�P   Au�uP   �|��Au���   Au�uP   Au��P   W�3rAu��   Au��P   Au�WP   ����Au�|�   Au�WP   Au��P   Vv��Au���   Au��P   Au�9P   ��.bAu�^�   Au�9P   Au��P   VHkIAu���   Au��P   Au�P   ��Au�@�   Au�P   Au��P   Wm0Au���   Au��P   Au��P   W��9Au�"�   Au��P   Au�nP   �Y�dAu���   Au�nP   Au��P   V�Au��   Au��P   Au�PP   WC��Au�u�   Au�PP   Au��P   �SҤAu���   Au��P   Au�2P   W]�Au�W�   Au�2P   AuʣP   V�q�Au���   AuʣP   Au�P   W�vAu�9�   Au�P   AuυP   V��AuϪ�   AuυP   Au��P   W!g�Au��   Au��P   Au�gP   ցԲAuԌ�   Au�gP   Au��P   VcbAu���   Au��P   Au�IP   W^�Au�n�   Au�IP   AuۺP   V!SAu���   AuۺP   Au�+P   U0� Au�P�   Au�+P   Au��P   W��iAu���   Au��P   Au�P   S���Au�2�   Au�P   Au�~P   W�@Au��   Au�~P   Au��P   U�DqAu��   Au��P   Au�`P   Wj�qAu��   Au�`P   Au��P   �V�6Au���   Au��P   Au�BP   W<�UAu�g�   Au�BP   Au�P   V�D�Au���   Au�P   Au�$P   ՚�$Au�I�   Au�$P   Au��P   W��Au���   Au��P   Au�P   U��Au�+�   Au�P   Au�wP   We�wAu���   Au�wP   Au��P   ����Au��   Au��P   Av YP   VЫAv ~�   Av YP   Av�P   V���Av��   Av�P   Av;P   W}��Av`�   Av;P   Av�P   �U�Av��   Av�P   Av
P   W9��Av
B�   Av
P   Av�P   ֋HAv��   Av�P   Av�P   W&�Av$�   Av�P   AvpP   �µ	Av��   AvpP   Av�P   V3d�Av�   Av�P   AvRP   V���Avw�   AvRP   Av�P   W~��Av��   Av�P   Av4P   ֒�@AvY�   Av4P   Av�P   W'�Av��   Av�P   Av P   �\hcAv ;�   Av P   Av"�P   W\�Av"��   Av"�P   Av$�P   V�prAv%�   Av$�P   Av'iP   V�{Av'��   Av'iP   Av)�P   V�*�Av)��   Av)�P   Av,KP   V��{Av,p�   Av,KP   Av.�P   ���Av.��   Av.�P   Av1-P   W>^Av1R�   Av1-P   Av3�P   WĿAv3��   Av3�P   Av6P   U��ZAv64�   Av6P   Av8�P   W4�NAv8��   Av8�P   Av:�P   ��'Av;�   Av:�P   Av=bP   Vs��Av=��   Av=bP   Av?�P   ��D�Av?��   Av?�P   AvBDP   WؕAvBi�   AvBDP   AvD�P   T��[AvD��   AvD�P   AvG&P   V��^AvGK�   AvG&P   AvI�P   W#X�AvI��   AvI�P   AvLP   We�|AvL-�   AvLP   AvNyP   ՀjAvN��   AvNyP   AvP�P   W2	tAvQ�   AvP�P   AvS[P   �=�AvS��   AvS[P   AvU�P   �xAvU��   AvU�P   AvX=P   ֔��AvXb�   AvX=P   AvZ�P   W�AvZ��   AvZ�P   Av]P   �ڄ�Av]D�   Av]P   Av_�P   �9�`Av_��   Av_�P   AvbP   �ZE�Avb&�   AvbP   AvdrP   U�I�Avd��   AvdrP   Avf�P   V��Avg�   Avf�P   AviTP   � /g