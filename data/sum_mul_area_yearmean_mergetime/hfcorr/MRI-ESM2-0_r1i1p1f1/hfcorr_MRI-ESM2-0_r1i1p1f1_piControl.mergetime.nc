CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         AJ�       creation_date         2019-12-10T11:45:46Z   data_specs_version        01.00.31   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.piControl.none.r1i1p1f1    grid      4native ocean tri-polar grid with 360x363 ocean cells   
grid_label        gn     history      &Tue May 30 16:59:34 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfcorr/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Omon.hfcorr.gn.v20210311/hfcorr_Omon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.yearmean.mul.areacello_piControl_v20191224.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfcorr/MRI-ESM2-0_r1i1p1f1/hfcorr_MRI-ESM2-0_r1i1p1f1_piControl.mergetime.nc
Fri Nov 04 02:07:51 2022: cdo -O -s -fldsum -setattribute,hfcorr@units=W m-2 m2 -mul -yearmean -selname,hfcorr /Users/benjamin/Data/p22b/CMIP6/hfcorr/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Omon.hfcorr.gn.v20210311/hfcorr_Omon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.nc /Users/benjamin/Data/p22b/CMIP6/areacello/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Ofx.areacello.gn.v20191224/areacello_Ofx_MRI-ESM2-0_piControl_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfcorr/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.piControl.r1i1p1f1.Omon.hfcorr.gn.v20210311/hfcorr_Omon_MRI-ESM2-0_piControl_r1i1p1f1_gn_185001-255012.yearmean.mul.areacello_piControl_v20191224.fldsum.nc
2019-12-10T11:45:46Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.     initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ACreation Date:(24 July 2019) MD5:c93735846d66458966fc81f390b2d714      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       hfcorr     variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.5.0      tracking_id       1hdl:21.14100/d665b5d7-f0e1-4699-9591-5aea0026d41b      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               $   	time_bnds                                 ,   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X                  lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y                  hfcorr                     	   standard_name         heat_flux_correction   	long_name         Heat Flux Correction   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       �Flux correction is also called 'flux adjustment'. A positive flux correction is downward i.e. added to the ocean. In accordance with common usage in geophysical disciplines, 'flux' implies per unit area, called 'flux density' in physics.      cell_measures         area: areacello    history       �2019-12-10T11:45:46Z altered by CMOR: replaced missing value flag (-9.99e+33) and corresponding data with standard missing value (1e+20).               <+20).               <                Aq���   Aq��P   Aq�P   Wd�Aq�6�   Aq�P   Aq��P   WK�Aq���   Aq��P   Aq��P   W
	Aq��   Aq��P   Aq�dP   W�Aq���   Aq�dP   Aq��P   Wk1Aq���   Aq��P   Aq�FP   V��[Aq�k�   Aq�FP   Aq��P   W(�Aq���   Aq��P   Aq�(P   W�Aq�M�   Aq�(P   Aq��P   Ww�Aq���   Aq��P   Aq�
P   W}�Aq�/�   Aq�
P   Aq�{P   W ��Aq���   Aq�{P   Aq��P   W�<Aq��   Aq��P   Aq�]P   W�AqĂ�   Aq�]P   Aq��P   W�&Aq���   Aq��P   Aq�?P   W�Aq�d�   Aq�?P   Aq˰P   W�tAq���   Aq˰P   Aq�!P   V�{EAq�F�   Aq�!P   AqВP   V�>bAqз�   AqВP   Aq�P   V�7Aq�(�   Aq�P   Aq�tP   W�Aqՙ�   Aq�tP   Aq��P   W�Aq�
�   Aq��P   Aq�VP   Wr.Aq�{�   Aq�VP   Aq��P   WbtAq���   Aq��P   Aq�8P   W42Aq�]�   Aq�8P   Aq�P   W��Aq���   Aq�P   Aq�P   V�"�Aq�?�   Aq�P   Aq�P   W*ZAq��   Aq�P   Aq��P   V�_)Aq�!�   Aq��P   Aq�mP   W�-Aq��   Aq�mP   Aq��P   WnBAq��   Aq��P   Aq�OP   W	��Aq�t�   Aq�OP   Aq��P   W @uAq���   Aq��P   Aq�1P   V�QAq�V�   Aq�1P   Aq��P   V�l�Aq���   Aq��P   Aq�P   W�HAq�8�   Aq�P   Aq��P   We�Aq���   Aq��P   Aq��P   W q$Aq��   Aq��P   ArfP   W��Ar��   ArfP   Ar�P   V���Ar��   Ar�P   ArHP   V��!Arm�   ArHP   Ar�P   W�ZAr��   Ar�P   Ar*P   WX�ArO�   Ar*P   Ar�P   W �&Ar��   Ar�P   ArP   V�4<Ar1�   ArP   Ar}P   W>Ar��   Ar}P   Ar�P   V�ZcAr�   Ar�P   Ar_P   WWAr��   Ar_P   Ar�P   Wd7Ar��   Ar�P   ArAP   W��Arf�   ArAP   Ar�P   W	�Ar��   Ar�P   Ar!#P   Wz�Ar!H�   Ar!#P   Ar#�P   W^Ar#��   Ar#�P   Ar&P   W	��Ar&*�   Ar&P   Ar(vP   V��Ar(��   Ar(vP   Ar*�P   V���Ar+�   Ar*�P   Ar-XP   W)�Ar-}�   Ar-XP   Ar/�P   W�*Ar/��   Ar/�P   Ar2:P   W�Ar2_�   Ar2:P   Ar4�P   W�
Ar4��   Ar4�P   Ar7P   W
vAr7A�   Ar7P   Ar9�P   W2aAr9��   Ar9�P   Ar;�P   Wy�Ar<#�   Ar;�P   Ar>oP   W\�Ar>��   Ar>oP   Ar@�P   V��vArA�   Ar@�P   ArCQP   V���ArCv�   ArCQP   ArE�P   V���ArE��   ArE�P   ArH3P   W`�ArHX�   ArH3P   ArJ�P   V��ArJ��   ArJ�P   ArMP   Wf�ArM:�   ArMP   ArO�P   W<�ArO��   ArO�P   ArQ�P   V��ArR�   ArQ�P   ArThP   W?�ArT��   ArThP   ArV�P   W��ArV��   ArV�P   ArYJP   W��ArYo�   ArYJP   Ar[�P   W�Ar[��   Ar[�P   Ar^,P   WE�Ar^Q�   Ar^,P   Ar`�P   W��Ar`��   Ar`�P   ArcP   W��Arc3�   ArcP   AreP   W�Are��   AreP   Arg�P   W��Arh�   Arg�P   ArjaP   W�OArj��   ArjaP   Arl�P   W4�Arl��   Arl�P   AroCP   W!�Aroh�   AroCP   Arq�P   W�;Arq��   Arq�P   Art%P   WF�ArtJ�   Art%P   Arv�P   W
7KArv��   Arv�P   AryP   W�7Ary,�   AryP   Ar{xP   W�Ar{��   Ar{xP   Ar}�P   W
��Ar~�   Ar}�P   Ar�ZP   W
�Ar��   Ar�ZP   Ar��P   Wy�Ar���   Ar��P   Ar�<P   W$cAr�a�   Ar�<P   Ar��P   W~!Ar���   Ar��P   Ar�P   W $aAr�C�   Ar�P   Ar��P   V���Ar���   Ar��P   Ar� P   W�Ar�%�   Ar� P   Ar�qP   W�,Ar���   Ar�qP   Ar��P   V��xAr��   Ar��P   Ar�SP   V��oAr�x�   Ar�SP   Ar��P   V�fAr���   Ar��P   Ar�5P   V�7Ar�Z�   Ar�5P   Ar��P   WïAr���   Ar��P   Ar�P   W��Ar�<�   Ar�P   Ar��P   V��-Ar���   Ar��P   Ar��P   W\Ar��   Ar��P   Ar�jP   V��Ar���   Ar�jP   Ar��P   W�Ar� �   Ar��P   Ar�LP   V�{�Ar�q�   Ar�LP   Ar��P   W ��Ar���   Ar��P   Ar�.P   WdtAr�S�   Ar�.P   Ar��P   W��Ar���   Ar��P   Ar�P   W��Ar�5�   Ar�P   Ar��P   W�Ar���   Ar��P   Ar��P   W �Ar��   Ar��P   Ar�cP   W�Ar���   Ar�cP   Ar��P   W|�Ar���   Ar��P   Ar�EP   W�5Ar�j�   Ar�EP   ArĶP   V�HKAr���   ArĶP   Ar�'P   W�Ar�L�   Ar�'P   ArɘP   V���Arɽ�   ArɘP   Ar�	P   V�9Ar�.�   Ar�	P   Ar�zP   V�R�ArΟ�   Ar�zP   Ar��P   W �{Ar��   Ar��P   Ar�\P   W�aArӁ�   Ar�\P   Ar��P   WCAr���   Ar��P   Ar�>P   W=�Ar�c�   Ar�>P   ArگP   W�_Ar���   ArگP   Ar� P   WW�Ar�E�   Ar� P   ArߑP   W �@Ar߶�   ArߑP   Ar�P   W�JAr�'�   Ar�P   Ar�sP   W��Ar��   Ar�sP   Ar��P   W0Ar�	�   Ar��P   Ar�UP   W��Ar�z�   Ar�UP   Ar��P   W ��Ar���   Ar��P   Ar�7P   V�5�Ar�\�   Ar�7P   Ar�P   V�c@Ar���   Ar�P   Ar�P   V�$Ar�>�   Ar�P   Ar��P   W	PAr���   Ar��P   Ar��P   V��Ar� �   Ar��P   Ar�lP   V�@pAr���   Ar�lP   Ar��P   W �@Ar��   Ar��P   Ar�NP   V��Ar�s�   Ar�NP   As�P   V�L�As��   As�P   As0P   V���AsU�   As0P   As�P   W��As��   As�P   As	P   W x�As	7�   As	P   As�P   V�v�As��   As�P   As�P   V�1As�   As�P   AseP   WmdAs��   AseP   As�P   W &�As��   As�P   AsGP   V��4Asl�   AsGP   As�P   V��YAs��   As�P   As)P   V�AsN�   As)P   As�P   V��As��   As�P   AsP   V�CXAs0�   AsP   As!|P   V��As!��   As!|P   As#�P   V���As$�   As#�P   As&^P   V�@�As&��   As&^P   As(�P   Wx�As(��   As(�P   As+@P   W��As+e�   As+@P   As-�P   W ��As-��   As-�P   As0"P   W��As0G�   As0"P   As2�P   Wu9As2��   As2�P   As5P   WPoAs5)�   As5P   As7uP   Wj�As7��   As7uP   As9�P   WW�As:�   As9�P   As<WP   V�ZDAs<|�   As<WP   As>�P   WqwAs>��   As>�P   AsA9P   V��AsA^�   AsA9P   AsC�P   WlAsC��   AsC�P   AsFP   W�ZAsF@�   AsFP   AsH�P   V�#AsH��   AsH�P   AsJ�P   V���AsK"�   AsJ�P   AsMnP   V�DGAsM��   AsMnP   AsO�P   W�gAsP�   AsO�P   AsRPP   W�AsRu�   AsRPP   AsT�P   W �AsT��   AsT�P   AsW2P   W�AsWW�   AsW2P   AsY�P   W��AsY��   AsY�P   As\P   WH%As\9�   As\P   As^�P   W :As^��   As^�P   As`�P   W�KAsa�   As`�P   AscgP   W ��Asc��   AscgP   Ase�P   W��Ase��   Ase�P   AshIP   W �Ashn�   AshIP   Asj�P   V�;�Asj��   Asj�P   Asm+P   V��ZAsmP�   Asm+P   Aso�P   V�eAso��   Aso�P   AsrP   W��Asr2�   AsrP   Ast~P   W3*Ast��   Ast~P   Asv�P   W�Asw�   Asv�P   Asy`P   W
r�Asy��   Asy`P   As{�P   W�jAs{��   As{�P   As~BP   W��As~g�   As~BP   As��P   Wd*As���   As��P   As�$P   V�w�As�I�   As�$P   As��P   W �As���   As��P   As�P   V���As�+�   As�P   As�wP   V�S�As���   As�wP   As��P   W�]As��   As��P   As�YP   V��As�~�   As�YP   As��P   W As���   As��P   As�;P   W �As�`�   As�;P   As��P   V���As���   As��P   As�P   W �As�B�   As�P   As��P   W?uAs���   As��P   As��P   W>_As�$�   As��P   As�pP   V��As���   As�pP   As��P   W �XAs��   As��P   As�RP   W=As�w�   As�RP   As��P   V��GAs���   As��P   As�4P   V��As�Y�   As�4P   As��P   V��As���   As��P   As�P   V�]>As�;�   As�P   As��P   V�&FAs���   As��P   As��P   V���As��   As��P   As�iP   W !XAs���   As�iP   As��P   V��As���   As��P   As�KP   V�}FAs�p�   As�KP   As��P   V�fAs���   As��P   As�-P   W:uAs�R�   As�-P   AsP   W��As���   AsP   As�P   WAs�4�   As�P   AsǀP   W4�Asǥ�   AsǀP   As��P   W��As��   As��P   As�bP   WQ�Aṡ�   As�bP   As��P   W/�As���   As��P   As�DP   W�As�i�   As�DP   AsӵP   W EiAs���   AsӵP   As�&P   W �As�K�   As�&P   AsؗP   W$Asؼ�   AsؗP   As�P   WGAs�-�   As�P   As�yP   W|�Asݞ�   As�yP   As��P   V�v�As��   As��P   As�[P   W N�As��   As�[P   As��P   V��+As���   As��P   As�=P   W:TAs�b�   As�=P   As�P   V�ҧAs���   As�P   As�P   V��As�D�   As�P   As�P   V�]�As��   As�P   As�P   V�4jAs�&�   As�P   As�rP   W��As��   As�rP   As��P   V�%�As��   As��P   As�TP   W �As�y�   As�TP   As��P   W�As���   As��P   As�6P   W �As�[�   As�6P   As��P   V�jTAs���   As��P   AtP   V��iAt=�   AtP   At�P   V���At��   At�P   At�P   W�,At�   At�P   At	kP   V��At	��   At	kP   At�P   W��At�   At�P   AtMP   W oAtr�   AtMP   At�P   Wc�At��   At�P   At/P   WdKAtT�   At/P   At�P   V�d�At��   At�P   AtP   V�ںAt6�   AtP   At�P   W;DAt��   At�P   At�P   V�66At�   At�P   AtdP   W�EAt��   AtdP   At!�P   W0&At!��   At!�P   At$FP   W��At$k�   At$FP   At&�P   W��At&��   At&�P   At)(P   W��At)M�   At)(P   At+�P   W��At+��   At+�P   At.
P   V�`�At./�   At.
P   At0{P   W�At0��   At0{P   At2�P   W7At3�   At2�P   At5]P   W�DAt5��   At5]P   At7�P   W�At7��   At7�P   At:?P   W.�At:d�   At:?P   At<�P   W �At<��   At<�P   At?!P   WsAt?F�   At?!P   AtA�P   W�AtA��   AtA�P   AtDP   V���AtD(�   AtDP   AtFtP   V��AtF��   AtFtP   AtH�P   W _fAtI
�   AtH�P   AtKVP   V�y�AtK{�   AtKVP   AtM�P   V�RQAtM��   AtM�P   AtP8P   V�8�AtP]�   AtP8P   AtR�P   V�AEAtR��   AtR�P   AtUP   WhoAtU?�   AtUP   AtW�P   V�"^AtW��   AtW�P   AtY�P   V�'AtZ!�   AtY�P   At\mP   Wq�At\��   At\mP   At^�P   W	�hAt_�   At^�P   AtaOP   W�|Atat�   AtaOP   Atc�P   W	RAtc��   Atc�P   Atf1P   W�tAtfV�   Atf1P   Ath�P   W�Ath��   Ath�P   AtkP   W��Atk8�   AtkP   Atm�P   V�x�Atm��   Atm�P   Ato�P   V� "Atp�   Ato�P   AtrfP   V��Atr��   AtrfP   Att�P   WB�Att��   Att�P   AtwHP   V��Atwm�   AtwHP   Aty�P   V�.Aty��   Aty�P   At|*P   V�\iAt|O�   At|*P   At~�P   W�EAt~��   At~�P   At�P   W0�At�1�   At�P   At�}P   W��At���   At�}P   At��P   W��At��   At��P   At�_P   W �,At���   At�_P   At��P   W\�At���   At��P   At�AP   W U�At�f�   At�AP   At��P   V�JUAt���   At��P   At�#P   W0�At�H�   At�#P   At��P   W4�At���   At��P   At�P   W,_At�*�   At�P   At�vP   W �cAt���   At�vP   At��P   Wv�At��   At��P   At�XP   V��.At�}�   At�XP   At��P   WgeAt���   At��P   At�:P   W��At�_�   At�:P   At��P   V��At���   At��P   At�P   V��At�A�   At�P   At��P   W�At���   At��P   At��P   WxEAt�#�   At��P   At�oP   WC�At���   At�oP   At��P   W|�At��   At��P   At�QP   W	SiAt�v�   At�QP   At��P   V��
At���   At��P   At�3P   V�[RAt�X�   At�3P   At��P   V�BwAt���   At��P   At�P   WY3At�:�   At�P   At��P   W�At���   At��P   At��P   W�At��   At��P   At�hP   V���Atō�   At�hP   At��P   W�At���   At��P   At�JP   V���At�o�   At�JP   At̻P   V��NAt���   At̻P   At�,P   W(`At�Q�   At�,P   AtѝP   W ��At���   AtѝP   At�P   V��0At�3�   At�P   At�P   V���At֤�   At�P   At��P   V���At��   At��P   At�aP   WzAtۆ�   At�aP   At��P   W��At���   At��P   At�CP   W
I�At�h�   At�CP   At�P   W	@�At���   At�P   At�%P   V���At�J�   At�%P   At�P   W�rAt��   At�P   At�P   V���At�,�   At�P   At�xP   V�%�At��   At�xP   At��P   V���At��   At��P   At�ZP   V�,�At��   At�ZP   At��P   V�r�At���   At��P   At�<P   V��UAt�a�   At�<P   At��P   W~�At���   At��P   At�P   V�C�At�C�   At�P   At��P   V��At���   At��P   Au  P   W��Au %�   Au  P   AuqP   V���Au��   AuqP   Au�P   V�^Au�   Au�P   AuSP   W\�Aux�   AuSP   Au	�P   W .Au	��   Au	�P   Au5P   Wr#AuZ�   Au5P   Au�P   W£Au��   Au�P   AuP   W��Au<�   AuP   Au�P   V�7.Au��   Au�P   Au�P   WAu�   Au�P   AujP   V��;Au��   AujP   Au�P   W��Au �   Au�P   AuLP   W h&Auq�   AuLP   Au�P   V��Au��   Au�P   Au".P   W]�Au"S�   Au".P   Au$�P   V�qAu$��   Au$�P   Au'P   WG�Au'5�   Au'P   Au)�P   W��Au)��   Au)�P   Au+�P   V���Au,�   Au+�P   Au.cP   W��Au.��   Au.cP   Au0�P   W�YAu0��   Au0�P   Au3EP   WT[Au3j�   Au3EP   Au5�P   Wp�Au5��   Au5�P   Au8'P   W�Au8L�   Au8'P   Au:�P   V��'Au:��   Au:�P   Au=	P   W��Au=.�   Au=	P   Au?zP   W�Au?��   Au?zP   AuA�P   W='AuB�   AuA�P   AuD\P   W{hAuD��   AuD\P   AuF�P   W1�AuF��   AuF�P   AuI>P   W^�AuIc�   AuI>P   AuK�P   W`�AuK��   AuK�P   AuN P   W��AuNE�   AuN P   AuP�P   W�AuP��   AuP�P   AuSP   V���AuS'�   AuSP   AuUsP   V��AuU��   AuUsP   AuW�P   V��:AuX	�   AuW�P   AuZUP   V��AuZz�   AuZUP   Au\�P   WAu\��   Au\�P   Au_7P   WuAu_\�   Au_7P   Aua�P   Wu�Aua��   Aua�P   AudP   V��Aud>�   AudP   Auf�P   W�Auf��   Auf�P   Auh�P   V��Aui �   Auh�P   AuklP   V�;~Auk��   AuklP   Aum�P   V�<qAun�   Aum�P   AupNP   V�k�Aups�   AupNP   Aur�P   WP'Aur��   Aur�P   Auu0P   V���AuuU�   Auu0P   Auw�P   V�x�Auw��   Auw�P   AuzP   V���Auz7�   AuzP   Au|�P   W��Au|��   Au|�P   Au~�P   V�^Au�   Au~�P   Au�eP   W ��Au���   Au�eP   Au��P   W
ncAu���   Au��P   Au�GP   W �Au�l�   Au�GP   Au��P   V�/�Au���   Au��P   Au�)P   W �Au�N�   Au�)P   Au��P   V��tAu���   Au��P   Au�P   V�<Au�0�   Au�P   Au�|P   W �Au���   Au�|P   Au��P   W 8�Au��   Au��P   Au�^P   W�CAu���   Au�^P   Au��P   W�Au���   Au��P   Au�@P   V�4GAu�e�   Au�@P   Au��P   W �Au���   Au��P   Au�"P   W�TAu�G�   Au�"P   Au��P   V�s�Au���   Au��P   Au�P   Wb�Au�)�   Au�P   Au�uP   V�ytAu���   Au�uP   Au��P   V�z?Au��   Au��P   Au�WP   W��Au�|�   Au�WP   Au��P   V�ΝAu���   Au��P   Au�9P   W"Au�^�   Au�9P   Au��P   V���Au���   Au��P   Au�P   V�,�Au�@�   Au�P   Au��P   W��Au���   Au��P   Au��P   V�rTAu�"�   Au��P   Au�nP   V�?�Au���   Au�nP   Au��P   V�_Au��   Au��P   Au�PP   V���Au�u�   Au�PP   Au��P   V�GAu���   Au��P   Au�2P   V�G'Au�W�   Au�2P   AuʣP   V���Au���   AuʣP   Au�P   V���Au�9�   Au�P   AuυP   W>�AuϪ�   AuυP   Au��P   V�RgAu��   Au��P   Au�gP   V���AuԌ�   Au�gP   Au��P   V�5�Au���   Au��P   Au�IP   V���Au�n�   Au�IP   AuۺP   V�ѨAu���   AuۺP   Au�+P   W¤Au�P�   Au�+P   Au��P   W�Au���   Au��P   Au�P   V���Au�2�   Au�P   Au�~P   V���Au��   Au�~P   Au��P   W��Au��   Au��P   Au�`P   Wg]Au��   Au�`P   Au��P   W��Au���   Au��P   Au�BP   V���Au�g�   Au�BP   Au�P   V�O�Au���   Au�P   Au�$P   V���Au�I�   Au�$P   Au��P   V��Au���   Au��P   Au�P   WvbAu�+�   Au�P   Au�wP   W�nAu���   Au�wP   Au��P   W�YAu��   Au��P   Av YP   V��Av ~�   Av YP   Av�P   Wk�Av��   Av�P   Av;P   V�b�Av`�   Av;P   Av�P   V�#eAv��   Av�P   Av
P   W�Av
B�   Av
P   Av�P   V�)Av��   Av�P   Av�P   V�X�Av$�   Av�P   AvpP   W ��Av��   AvpP   Av�P   W�CAv�   Av�P   AvRP   W�XAvw�   AvRP   Av�P   W��Av��   Av�P   Av4P   V�JAvY�   Av4P   Av�P   W��Av��   Av�P   Av P   WdEAv ;�   Av P   Av"�P   W =OAv"��   Av"�P   Av$�P   V���Av%�   Av$�P   Av'iP   V�XgAv'��   Av'iP   Av)�P   V��dAv)��   Av)�P   Av,KP   W�Av,p�   Av,KP   Av.�P   V��Av.��   Av.�P   Av1-P   W�ZAv1R�   Av1-P   Av3�P   W��Av3��   Av3�P   Av6P   W,�Av64�   Av6P   Av8�P   V�+�Av8��   Av8�P   Av:�P   W��Av;�   Av:�P   Av=bP   W N%Av=��   Av=bP   Av?�P   W0�Av?��   Av?�P   AvBDP   W�^AvBi�   AvBDP   AvD�P   W �PAvD��   AvD�P   AvG&P   V�^�AvGK�   AvG&P   AvI�P   W4�AvI��   AvI�P   AvLP   V���AvL-�   AvLP   AvNyP   W �yAvN��   AvNyP   AvP�P   W$AvQ�   AvP�P   AvS[P   W�AvS��   AvS[P   AvU�P   W ��AvU��   AvU�P   AvX=P   W��AvXb�   AvX=P   AvZ�P   Wt�AvZ��   AvZ�P   Av]P   W V�Av]D�   Av]P   Av_�P   V�*�Av_��   Av_�P   AvbP   V�-�Avb&�   AvbP   AvdrP   W��Avd��   AvdrP   Avf�P   Wy&Avg�   Avf�P   AviTP   W��Aviy�   AviTP   Avk�P   WD�Avk��   Avk�P   Avn6P   W�Avn[�   Avn6P   Avp�P   Wv�Avp��   Avp�P   AvsP   WV'Avs=�   AvsP   Avu�P   V��Avu��   Avu�P   Avw�P   W z�Avx�   Avw�P   AvzkP   V��Avz��   AvzkP   Av|�P   W��Av}�   Av|�P   AvMP   W��Avr�   AvMP   Av��P   V��Av���   Av��P   Av�/P   V��yAv�T�   Av�/P   Av��P   W�LAv���   Av��P   Av�P   W�uAv�6�   Av�P   Av��P   V�2�Av���   Av��P   Av��P   WcAv��   Av��P   Av�dP   V���Av���   Av�dP   Av��P   V��Av���   Av��P   Av�FP   W>�Av�k�   Av�FP   Av��P   W �$Av���   Av��P   Av�(P   W�Av�M�   Av�(P   Av��P   W�Av���   Av��P   Av�
P   V���Av�/�   Av�
P   Av�{P   W �eAv���   Av�{P   Av��P   V���Av��   Av��P   Av�]P   V�Av���   Av�]P   Av��P   W �Av���   Av��P   Av�?P   V��Av�d�   Av�?P   Av��P   V��RAv���   Av��P   Av�!P   V��3Av�F�   Av�!P   Av��P   V��Av���   Av��P   Av�P   V�j�Av�(�   Av�P   Av�tP   V�o'Av���   Av�tP   Av��P   V��Av�
�   Av��P   Av�VP   V���Av�{�   Av�VP   Av��P   V��(Av���   Av��P   Av�8P   W2�Av�]�   Av�8P   AvéP   W��Av���   AvéP   Av�P   W ��Av�?�   Av�P   AvȋP   W��AvȰ�   AvȋP   Av��P   V���Av�!�   Av��P   Av�mP   V���Av͒�   Av�mP   Av��P   WV�Av��   Av��P   Av�OP   V�AAv�t�   Av�OP   Av��P   V�OQAv���   Av��P   Av�1P   W �,Av�V�   Av�1P   Av٢P   W�PAv���   Av٢P   Av�P   V�EAv�8�   Av�P   AvބP   W�Avީ�   AvބP   Av��P   V�%�Av��   Av��P   Av�fP   W�MAv��   Av�fP   Av��P   W��Av���   Av��P   Av�HP   W �qAv�m�   Av�HP   Av�P   V���Av���   Av�P   Av�*P   WtKAv�O�   Av�*P   Av�P   V�L�Av���   Av�P   Av�P   WӀAv�1�   Av�P   Av�}P   W�MAv���   Av�}P   Av��P   WԁAv��   Av��P   Av�_P   V��=Av���   Av�_P   Av��P   V��Av���   Av��P   Av�AP   V�c�Av�f�   Av�AP   Aw �P   W �
Aw ��   Aw �P   Aw#P   V��AwH�   Aw#P   Aw�P   W�tAw��   Aw�P   AwP   WP'Aw*�   AwP   Aw
vP   W �Aw
��   Aw
vP   Aw�P   W��Aw�   Aw�P   AwXP   Ww�Aw}�   AwXP   Aw�P   V�Aw��   Aw�P   Aw:P   V���Aw_�   Aw:P   Aw�P   W~�Aw��   Aw�P   AwP   W��AwA�   AwP   Aw�P   W�TAw��   Aw�P   Aw�P   V�$jAw#�   Aw�P   Aw oP   V�Aw ��   Aw oP   Aw"�P   V��Aw#�   Aw"�P   Aw%QP   V���Aw%v�   Aw%QP   Aw'�P   V��(Aw'��   Aw'�P   Aw*3P   V�{Aw*X�   Aw*3P   Aw,�P   W Aw,��   Aw,�P   Aw/P   W_#Aw/:�   Aw/P   Aw1�P   V��Aw1��   Aw1�P   Aw3�P   V�1�Aw4�   Aw3�P   Aw6hP   V�Aw6��   Aw6hP   Aw8�P   WtjAw8��   Aw8�P   Aw;JP   W n�Aw;o�   Aw;JP   Aw=�P   V�5`Aw=��   Aw=�P   Aw@,P   V��_Aw@Q�   Aw@,P   AwB�P   V��dAwB��   AwB�P   AwEP   V�X(AwE3�   AwEP   AwGP   V��nAwG��   AwGP   AwI�P   V�b&AwJ�   AwI�P   AwLaP   W��AwL��   AwLaP   AwN�P   V�� AwN��   AwN�P   AwQCP   V�F
AwQh�   AwQCP   AwS�P   W�eAwS��   AwS�P   AwV%P   W
AwVJ�   AwV%P   AwX�P   WɭAwX��   AwX�P   Aw[P   W ��Aw[,�   Aw[P   Aw]xP   V�D�Aw]��   Aw]xP   Aw_�P   WAyAw`�   Aw_�P   AwbZP   W7/Awb�   AwbZP   Awd�P   V���Awd��   Awd�P   Awg<P   V�5Awga�   Awg<P   Awi�P   V��Awi��   Awi�P   AwlP   W 2�AwlC�   AwlP   Awn�P   V��KAwn��   Awn�P   Awq P   W֦Awq%�   Awq P   AwsqP   W��Aws��   AwsqP   Awu�P   W��Awv�   Awu�P   AwxSP   W_Awxx�   AwxSP   Awz�P   V�s4Awz��   Awz�P   Aw}5P   V��3Aw}Z�   Aw}5P   Aw�P   W&�Aw��   Aw�P   Aw�P   V��Aw�<�   Aw�P   Aw��P   V�ydAw���   Aw��P   Aw��P   V��Aw��   Aw��P   Aw�jP   V�u	Aw���   Aw�jP   Aw��P   V�f�Aw� �   Aw��P   Aw�LP   V�Aw�q�   Aw�LP   Aw��P   V�ɎAw���   Aw��P   Aw�.P   W�kAw�S�   Aw�.P   Aw��P   W �Aw���   Aw��P   Aw�P   V�E�Aw�5�   Aw�P   Aw��P   V���Aw���   Aw��P   Aw��P   W9�Aw��   Aw��P   Aw�cP   W��Aw���   Aw�cP   Aw��P   WA�Aw���   Aw��P   Aw�EP   W5ZAw�j�   Aw�EP   Aw��P   Wp�Aw���   Aw��P   Aw�'P   WD�Aw�L�   Aw�'P   Aw��P   W�3Aw���   Aw��P   Aw�	P   W�Aw�.�   Aw�	P   Aw�zP   V�5�Aw���   Aw�zP   Aw��P   W ��Aw��   Aw��P   Aw�\P   W::Aw���   Aw�\P   Aw��P   WX}Aw���   Aw��P   Aw�>P   W��Aw�c�   Aw�>P   Aw��P   W�Aw���   Aw��P   Aw� P   V�߃Aw�E�   Aw� P   Aw��P   W\qAw���   Aw��P   Aw�P   V��+Aw�'�   Aw�P   Aw�sP   W ;�AwƘ�   Aw�sP   Aw��P   V��BAw�	�   Aw��P   Aw�UP   V�EAw�z�   Aw�UP   Aw��P   W0Aw���   Aw��P   Aw�7P   WxhAw�\�   Aw�7P   AwҨP   W�9Aw���   AwҨP   Aw�P   V��Aw�>�   Aw�P   Aw׊P   W?�Awׯ�   Aw׊P   Aw��P   W4�Aw� �   Aw��P   Aw�lP   W�Awܑ�   Aw�lP   Aw��P   W�Aw��   Aw��P   Aw�NP   W��Aw�s�   Aw�NP   Aw�P   W Z5Aw���   Aw�P   Aw�0P   Wk/Aw�U�   Aw�0P   Aw�P   W �@Aw���   Aw�P   Aw�P   W �Aw�7�   Aw�P   Aw�P   V�	Aw���   Aw�P   Aw��P   V�~BAw��   Aw��P   Aw�eP   W �LAw��   Aw�eP   Aw��P   V���Aw���   Aw��P   Aw�GP   W�Aw�l�   Aw�GP   Aw��P   W̖Aw���   Aw��P   Aw�)P   V���Aw�N�   Aw�)P   Aw��P   V�n#Aw���   Aw��P   AxP   W�MAx0�   AxP   Ax|P   V�:Ax��   Ax|P   Ax�P   WA�Ax�   Ax�P   Ax^P   WRZAx��   Ax^P   Ax
�P   WwAx
��   Ax
�P   Ax@P   W3�Axe�   Ax@P   Ax�P   W �Ax��   Ax�P   Ax"P   W��AxG�   Ax"P   Ax�P   V�Z�Ax��   Ax�P   AxP   W��Ax)�   AxP   AxuP   W =)Ax��   AxuP   Ax�P   W:�Ax�   Ax�P   AxWP   WݱAx|�   AxWP   Ax �P   V�jAx ��   Ax �P   Ax#9P   W�Ax#^�   Ax#9P   Ax%�P   W}�Ax%��   Ax%�P   Ax(P   W ,YAx(@�   Ax(P   Ax*�P   W ��Ax*��   Ax*�P   Ax,�P   W�iAx-"�   Ax,�P   Ax/nP   W�?Ax/��   Ax/nP   Ax1�P   W�7Ax2�   Ax1�P   Ax4PP   W �;Ax4u�   Ax4PP   Ax6�P   V��9Ax6��   Ax6�P   Ax92P   W8�Ax9W�   Ax92P   Ax;�P   V�_�Ax;��   Ax;�P   Ax>P   W��Ax>9�   Ax>P   Ax@�P   V�C�Ax@��   Ax@�P   AxB�P   V���AxC�   AxB�P   AxEgP   V�9<AxE��   AxEgP   AxG�P   WAxG��   AxG�P   AxJIP   V�M�AxJn�   AxJIP   AxL�P   V���AxL��   AxL�P   AxO+P   Wa�AxOP�   AxO+P   AxQ�P   V�CQAxQ��   AxQ�P   AxTP   V�E�