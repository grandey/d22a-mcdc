CDF  �   
      time       bnds      lon       lat          2   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �CMCC-ESM2 (2017): 
aerosol: MAM3
atmos: CAM5.3 (1deg; 288 x 192 longitude/latitude; 30 levels; top at ~2 hPa)
atmosChem: none
land: CLM4.5 (BGC mode)
landIce: none
ocean: NEMO3.6 (ORCA1 tripolar primarly 1 deg lat/lon with meridional refinement down to 1/3 degree in the tropics; 362 x 292 longitude/latitude; 50 vertical levels; top grid cell 0-1 m)
ocnBgchem: BFM5.1
seaIce: CICE4.0   institution       QFondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    comment       none   contact       	T. Lovato      creation_date         2021-03-03T08:47:08Z   data_specs_version        01.00.31   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Khttps://furtherinfo.es-doc.org/CMIP6.CMCC.CMCC-ESM2.piControl.none.r1i1p1f1    grid      native atmosphere regular grid     
grid_label        gn     history      Tue May 30 16:58:49 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.piControl.r1i1p1f1.Amon.rlut.gn.v20210304/rlut_Amon_CMCC-ESM2_piControl_r1i1p1f1_gn_185001-209912.yearmean.mul.areacella_piControl_v20210304.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.piControl.r1i1p1f1.Amon.rlut.gn.v20210304/rlut_Amon_CMCC-ESM2_piControl_r1i1p1f1_gn_210001-234912.yearmean.mul.areacella_piControl_v20210304.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/CMCC-ESM2_r1i1p1f1/rlut_CMCC-ESM2_r1i1p1f1_piControl.mergetime.nc
Fri Nov 04 02:57:49 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.piControl.r1i1p1f1.Amon.rlut.gn.v20210304/rlut_Amon_CMCC-ESM2_piControl_r1i1p1f1_gn_185001-209912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.piControl.r1i1p1f1.fx.areacella.gn.v20210304/areacella_fx_CMCC-ESM2_piControl_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CMCC-ESM2_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-ESM2.piControl.r1i1p1f1.Amon.rlut.gn.v20210304/rlut_Amon_CMCC-ESM2_piControl_r1i1p1f1_gn_185001-209912.yearmean.mul.areacella_piControl_v20210304.fldsum.nc
2021-03-03T08:47:08Z ;rewrote data to be consistent with CMIP for variable rlut found in table Amon.;
none     initialization_index            institution_id        CMCC   mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      	CMCC-ESM2      parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      
references        none   run_variant       1st realization    	source_id         	CMCC-ESM2      source_type       	AOGCM BGC      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(05 February 2020) MD5:6a248fd76c55aa6d6f7b3cc6866b5faf      title         #CMCC-ESM2 output prepared for CMIP6    variable_id       rlut   variant_label         r1i1p1f1   license      ?CMIP6 model data produced by CMCC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.6.0      tracking_id       1hdl:21.14100/a7e673bf-2ea1-4935-9246-7c3404a51414      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                   	   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         FLUT   cell_measures         area: areacella             �eacella             �                Aq���   Aq��P   Aq�P   [�;�Aq�6�   Aq�P   Aq��P   [�aAq���   Aq��P   Aq��P   [ٕLAq��   Aq��P   Aq�dP   [ٔ~Aq���   Aq�dP   Aq��P   [�i�Aq���   Aq��P   Aq�FP   [ڌ�Aq�k�   Aq�FP   Aq��P   [��tAq���   Aq��P   Aq�(P   [���Aq�M�   Aq�(P   Aq��P   [��SAq���   Aq��P   Aq�
P   [��Aq�/�   Aq�
P   Aq�{P   [ډyAq���   Aq�{P   Aq��P   [�?uAq��   Aq��P   Aq�]P   [��vAqĂ�   Aq�]P   Aq��P   [���Aq���   Aq��P   Aq�?P   [�DuAq�d�   Aq�?P   Aq˰P   [�g$Aq���   Aq˰P   Aq�!P   [ٻ�Aq�F�   Aq�!P   AqВP   [�r�Aqз�   AqВP   Aq�P   [���Aq�(�   Aq�P   Aq�tP   [ځ�Aqՙ�   Aq�tP   Aq��P   [�JCAq�
�   Aq��P   Aq�VP   [١�Aq�{�   Aq�VP   Aq��P   [٠�Aq���   Aq��P   Aq�8P   [��Aq�]�   Aq�8P   Aq�P   [��Aq���   Aq�P   Aq�P   [�|Aq�?�   Aq�P   Aq�P   [�mUAq��   Aq�P   Aq��P   [ٟ�Aq�!�   Aq��P   Aq�mP   [���Aq��   Aq�mP   Aq��P   [���Aq��   Aq��P   Aq�OP   [ٖ�Aq�t�   Aq�OP   Aq��P   [�3�Aq���   Aq��P   Aq�1P   [�bAq�V�   Aq�1P   Aq��P   [ٌ�Aq���   Aq��P   Aq�P   [��Aq�8�   Aq�P   Aq��P   [�%=Aq���   Aq��P   Aq��P   [�5�Aq��   Aq��P   ArfP   [�<>Ar��   ArfP   Ar�P   [�T�Ar��   Ar�P   ArHP   [�=�Arm�   ArHP   Ar�P   [�e|Ar��   Ar�P   Ar*P   [ٯArO�   Ar*P   Ar�P   [��Ar��   Ar�P   ArP   [ٲ�Ar1�   ArP   Ar}P   [�R�Ar��   Ar}P   Ar�P   [�^�Ar�   Ar�P   Ar_P   [�s�Ar��   Ar_P   Ar�P   [�^�Ar��   Ar�P   ArAP   [١�Arf�   ArAP   Ar�P   [���Ar��   Ar�P   Ar!#P   [٩�Ar!H�   Ar!#P   Ar#�P   [�>>Ar#��   Ar#�P   Ar&P   [�SPAr&*�   Ar&P   Ar(vP   [��JAr(��   Ar(vP   Ar*�P   [ٽ�Ar+�   Ar*�P   Ar-XP   [نAAr-}�   Ar-XP   Ar/�P   [ٯpAr/��   Ar/�P   Ar2:P   [٫vAr2_�   Ar2:P   Ar4�P   [�Ar4��   Ar4�P   Ar7P   [���Ar7A�   Ar7P   Ar9�P   [��mAr9��   Ar9�P   Ar;�P   [��jAr<#�   Ar;�P   Ar>oP   [كYAr>��   Ar>oP   Ar@�P   [٬�ArA�   Ar@�P   ArCQP   [��ArCv�   ArCQP   ArE�P   [�
�ArE��   ArE�P   ArH3P   [�|�ArHX�   ArH3P   ArJ�P   [�x�ArJ��   ArJ�P   ArMP   [�%�ArM:�   ArMP   ArO�P   [�(7ArO��   ArO�P   ArQ�P   [�u+ArR�   ArQ�P   ArThP   [���ArT��   ArThP   ArV�P   [�5ArV��   ArV�P   ArYJP   [ك ArYo�   ArYJP   Ar[�P   [�1�Ar[��   Ar[�P   Ar^,P   [�~�Ar^Q�   Ar^,P   Ar`�P   [ُ�Ar`��   Ar`�P   ArcP   [�h Arc3�   ArcP   AreP   [�\2Are��   AreP   Arg�P   [�v�Arh�   Arg�P   ArjaP   [�ҺArj��   ArjaP   Arl�P   [���Arl��   Arl�P   AroCP   [�}�Aroh�   AroCP   Arq�P   [�)xArq��   Arq�P   Art%P   [�v�ArtJ�   Art%P   Arv�P   [٭lArv��   Arv�P   AryP   [���Ary,�   AryP   Ar{xP   [��Ar{��   Ar{xP   Ar}�P   [��)Ar~�   Ar}�P   Ar�ZP   [پ�Ar��   Ar�ZP   Ar��P   [��xAr���   Ar��P   Ar�<P   [�ÐAr�a�   Ar�<P   Ar��P   [��Ar���   Ar��P   Ar�P   [���Ar�C�   Ar�P   Ar��P   [ټ�Ar���   Ar��P   Ar� P   [ّ�Ar�%�   Ar� P   Ar�qP   [���Ar���   Ar�qP   Ar��P   [ٖ�Ar��   Ar��P   Ar�SP   [�o�Ar�x�   Ar�SP   Ar��P   [ك�Ar���   Ar��P   Ar�5P   [��hAr�Z�   Ar�5P   Ar��P   [ِ=Ar���   Ar��P   Ar�P   [���Ar�<�   Ar�P   Ar��P   [�ȔAr���   Ar��P   Ar��P   [ى%Ar��   Ar��P   Ar�jP   [ٿ�Ar���   Ar�jP   Ar��P   [�wAr� �   Ar��P   Ar�LP   [ٮAr�q�   Ar�LP   Ar��P   [ً<Ar���   Ar��P   Ar�.P   [٩�Ar�S�   Ar�.P   Ar��P   [���Ar���   Ar��P   Ar�P   [��)Ar�5�   Ar�P   Ar��P   [�T�Ar���   Ar��P   Ar��P   [�3OAr��   Ar��P   Ar�cP   [�T�Ar���   Ar�cP   Ar��P   [ِ�Ar���   Ar��P   Ar�EP   [���Ar�j�   Ar�EP   ArĶP   [�:�Ar���   ArĶP   Ar�'P   [�U�Ar�L�   Ar�'P   ArɘP   [�W4Arɽ�   ArɘP   Ar�	P   [�Ar�.�   Ar�	P   Ar�zP   [��ArΟ�   Ar�zP   Ar��P   [�&PAr��   Ar��P   Ar�\P   [�9�ArӁ�   Ar�\P   Ar��P   [�cAr���   Ar��P   Ar�>P   [��Ar�c�   Ar�>P   ArگP   [ٳYAr���   ArگP   Ar� P   [�h�Ar�E�   Ar� P   ArߑP   [يAr߶�   ArߑP   Ar�P   [�kAr�'�   Ar�P   Ar�sP   [ٕAr��   Ar�sP   Ar��P   [ٸ�Ar�	�   Ar��P   Ar�UP   [�|�Ar�z�   Ar�UP   Ar��P   [�?UAr���   Ar��P   Ar�7P   [ق�Ar�\�   Ar�7P   Ar�P   [�B�Ar���   Ar�P   Ar�P   [�D�Ar�>�   Ar�P   Ar��P   [�r�Ar���   Ar��P   Ar��P   [�U�Ar� �   Ar��P   Ar�lP   [�V=Ar���   Ar�lP   Ar��P   [٠�Ar��   Ar��P   Ar�NP   [��kAr�s�   Ar�NP   As�P   [�QAs��   As�P   As0P   [ك�AsU�   As0P   As�P   [�}-As��   As�P   As	P   [ټ>As	7�   As	P   As�P   [���As��   As�P   As�P   [ڜ�As�   As�P   AseP   [ٖ|As��   AseP   As�P   [�nwAs��   As�P   AsGP   [��xAsl�   AsGP   As�P   [�PAs��   As�P   As)P   [�'QAsN�   As)P   As�P   [���As��   As�P   AsP   [�h�As0�   AsP   As!|P   [�fjAs!��   As!|P   As#�P   [ً�As$�   As#�P   As&^P   [��As&��   As&^P   As(�P   [ٗ�As(��   As(�P   As+@P   [��As+e�   As+@P   As-�P   [�%*As-��   As-�P   As0"P   [ڡEAs0G�   As0"P   As2�P   [�`�As2��   As2�P   As5P   [�m�As5)�   As5P   As7uP   [ّ�As7��   As7uP   As9�P   [��SAs:�   As9�P   As<WP   [��(As<|�   As<WP   As>�P   [�NAs>��   As>�P   AsA9P   [�:�AsA^�   AsA9P   AsC�P   [�@�AsC��   AsC�P   AsFP   [پ�AsF@�   AsFP   AsH�P   [�)nAsH��   AsH�P   AsJ�P   [��AsK"�   AsJ�P   AsMnP   [��AsM��   AsMnP   AsO�P   [�8�AsP�   AsO�P   AsRPP   [���AsRu�   AsRPP   AsT�P   [ٍ�AsT��   AsT�P   AsW2P   [م�AsWW�   AsW2P   AsY�P   [�8AsY��   AsY�P   As\P   [��&As\9�   As\P   As^�P   [ٍAs^��   As^�P   As`�P   [�<7Asa�   As`�P   AscgP   [�i�Asc��   AscgP   Ase�P   [ً�Ase��   Ase�P   AshIP   [�DxAshn�   AshIP   Asj�P   [�z�Asj��   Asj�P   Asm+P   [���AsmP�   Asm+P   Aso�P   [ٴ�Aso��   Aso�P   AsrP   [٫Asr2�   AsrP   Ast~P   [��Ast��   Ast~P   Asv�P   [ٲ�Asw�   Asv�P   Asy`P   [�-vAsy��   Asy`P   As{�P   [�+As{��   As{�P   As~BP   [ف�As~g�   As~BP   As��P   [ټ�As���   As��P   As�$P   [��0As�I�   As�$P   As��P   [���As���   As��P   As�P   [�v=As�+�   As�P   As�wP   [��As���   As�wP   As��P   [��As��   As��P   As�YP   [٨�As�~�   As�YP   As��P   [ف�As���   As��P   As�;P   [�H�As�`�   As�;P   As��P   [ق�As���   As��P   As�P   [ٻ�As�B�   As�P   As��P   [���As���   As��P   As��P   [ٵ+As�$�   As��P   As�pP   [��6As���   As�pP   As��P   [�>VAs��   As��P   As�RP   [�'�As�w�   As�RP   As��P   [٥�As���   As��P   As�4P   [نAs�Y�   As�4P   As��P   [ن�As���   As��P   As�P   [��|As�;�   As�P   As��P   [�6As���   As��P   As��P   [ڜ[As��   As��P   As�iP   [�B�As���   As�iP   As��P   [���As���   As��P   As�KP   [ټ�As�p�   As�KP   As��P   [ٽ�As���   As��P   As�-P   [���As�R�   As�-P   AsP   [�AAs���   AsP   As�P   [��mAs�4�   As�P   AsǀP   [ځAsǥ�   AsǀP   As��P   [�ݕAs��   As��P   As�bP   [��	Aṡ�   As�bP   As��P   [��As���   As��P   As�DP   [��As�i�   As�DP   AsӵP   [�&�As���   AsӵP   As�&P   [ګ�As�K�   As�&P   AsؗP   [��3Asؼ�   AsؗP   As�P   [���As�-�   As�P   As�yP   [ْ9Asݞ�   As�yP   As��P   [�$�As��   As��P   As�[P   [�XKAs��   As�[P   As��P   [��As���   As��P   As�=P   [ٜ�As�b�   As�=P   As�P   [���As���   As�P   As�P   [�As�D�   As�P   As�P   [�e�As��   As�P   As�P   [�%VAs�&�   As�P   As�rP   [�.�As��   As�rP   As��P   [��~As��   As��P   As�TP   [��nAs�y�   As�TP   As��P   [�
YAs���   As��P   As�6P   [��As�[�   As�6P   As��P   [٣�As���   As��P   AtP   [�d�At=�   AtP   At�P   [لAt��   At�P   At�P   [٦QAt�   At�P   At	kP   [��At	��   At	kP   At�P   [ڐ�At�   At�P   AtMP   [���Atr�   AtMP   At�P   [�SXAt��   At�P   At/P   [ٮKAtT�   At/P   At�P   [�	�At��   At�P   AtP   [��]At6�   AtP   At�P   [٥mAt��   At�P   At�P   [�oZAt�   At�P   AtdP   [��}At��   AtdP   At!�P   [��(At!��   At!�P   At$FP   [ٶsAt$k�   At$FP   At&�P   [ٞ�At&��   At&�P   At)(P   [ٮ�At)M�   At)(P   At+�P   [ٯ.At+��   At+�P   At.
P   [�tAt./�   At.
P   At0{P   [�D�At0��   At0{P   At2�P   [ى�At3�   At2�P   At5]P   [پ�At5��   At5]P   At7�P   [ٌ�At7��   At7�P   At:?P   [�xsAt:d�   At:?P   At<�P   [�Z�At<��   At<�P   At?!P   [� At?F�   At?!P   AtA�P   [��AtA��   AtA�P   AtDP   [�w�AtD(�   AtDP   AtFtP   [��AtF��   AtFtP   AtH�P   [�NgAtI
�   AtH�P   AtKVP   [ٜCAtK{�   AtKVP   AtM�P   [�8AtM��   AtM�P   AtP8P   [�I:AtP]�   AtP8P   AtR�P   [��AtR��   AtR�P   AtUP   [��AtU?�   AtUP   AtW�P   [�o�AtW��   AtW�P   AtY�P   [�7AtZ!�   AtY�P   At\mP   [�7�At\��   At\mP   At^�P   [ى�At_�   At^�P   AtaOP   [�{Atat�   AtaOP   Atc�P   [��Atc��   Atc�P   Atf1P   [�S�AtfV�   Atf1P   Ath�P   [��~Ath��   Ath�P   AtkP   [���Atk8�   AtkP   Atm�P   [�'uAtm��   Atm�P   Ato�P   [�{�Atp�   Ato�P   AtrfP   [�Atr��   AtrfP   Att�P   [لSAtt��   Att�P   AtwHP   [ٴ�Atwm�   AtwHP   Aty�P   [�P�Aty��   Aty�P   At|*P   [�6�At|O�   At|*P   At~�P   [�6�At~��   At~�P   At�P   [�~�At�1�   At�P   At�}P   [�KAt���   At�}P   At��P   [��At��   At��P   At�_P   [��7At���   At�_P   At��P   [�"�At���   At��P   At�AP   [�	At�f�   At�AP   At��P   [�uAt���   At��P   At�#P   [�99At�H�   At�#P   At��P   [ٱ�At���   At��P   At�P   [��rAt�*�   At�P   At�vP   [��At���   At�vP   At��P   [�|�At��   At��P   At�XP   [��TAt�}�   At�XP   At��P   [��"At���   At��P   At�:P   [�1�At�_�   At�:P   At��P   [ْ�At���   At��P   At�P   [���At�A�   At�P   At��P   [��At���   At��P   At��P   [��At�#�   At��P   At�oP   [�C�At���   At�oP   At��P   [�(7At��   At��P   At�QP   [��|At�v�   At�QP   At��P   [�qAt���   At��P   At�3P   [٩>At�X�   At�3P   At��P   [ٓ�At���   At��P   At�P   [�t�At�:�   At�P   At��P   [���At���   At��P   At��P   [ٲ�At��   At��P   At�hP   [٫~Atō�   At�hP   At��P   [�[At���   At��P   At�JP   [��3At�o�   At�JP   At̻P   [ٵ,At���   At̻P   At�,P   [�S�At�Q�   At�,P   AtѝP   [�]�At���   AtѝP   At�P   [ٮAt�3�   At�P   At�P   [� HAt֤�   At�P   At��P   [�=XAt��   At��P   At�aP   [ػfAtۆ�   At�aP   At��P   [خ�At���   At��P   At�CP   [�s�At�h�   At�CP   At�P   [���At���   At�P   At�%P   [��,At�J�   At�%P   At�P   [�o�At��   At�P   At�P   [��At�,�   At�P   At�xP   [��At��   At�xP   At��P   [�EkAt��   At��P   At�ZP   [�At��   At�ZP   At��P   [�i�At���   At��P   At�<P   [ِ�At�a�   At�<P   At��P   [�At���   At��P   At�P   [�dAt�C�   At�P   At��P   [�B;At���   At��P   Au  P   [�]Au %�   Au  P   AuqP   [�TAAu��   AuqP   Au�P   [� 9Au�   Au�P   AuSP   [�-Aux�   AuSP   Au	�P   [�&Au	��   Au	�P   Au5P   [�iAuZ�   Au5P   Au�P   [�w�Au��   Au�P   AuP   [�o�Au<�   AuP   Au�P   [�ǪAu��   Au�P   Au�P   [��2Au�   Au�P   AujP   [�<Au��   AujP   Au�P   [�aAu �   Au�P   AuLP   [�2�Auq�   AuLP   Au�P   [ڄ�Au��   Au�P   Au".P   [�,�Au"S�   Au".P   Au$�P   [�QAu$��   Au$�P   Au'P   [��Au'5�   Au'P   Au)�P   [���Au)��   Au)�P   Au+�P   [�VAu,�   Au+�P   Au.cP   [�֭Au.��   Au.cP   Au0�P   [�4Au0��   Au0�P   Au3EP   [���Au3j�   Au3EP   Au5�P   [ڧ!Au5��   Au5�P   Au8'P   [�e�Au8L�   Au8'P   Au:�P   [��UAu:��   Au:�P   Au=	P   [���Au=.�   Au=	P   Au?zP   [��Au?��   Au?zP   AuA�P   [�ofAuB�   AuA�P   AuD\P   [���AuD��   AuD\P   AuF�P   [�coAuF��   AuF�P   AuI>P   [ٿ�AuIc�   AuI>P   AuK�P   [��\AuK��   AuK�P   AuN P   [�.�AuNE�   AuN P   AuP�P   [��AuP��   AuP�P   AuSP   [�}�AuS'�   AuSP   AuUsP   [���AuU��   AuUsP   AuW�P   [���AuX	�   AuW�P   AuZUP   [��AuZz�   AuZUP   Au\�P   [�ߑAu\��   Au\�P   Au_7P   [�$wAu_\�   Au_7P   Aua�P   [�iAua��   Aua�P   AudP   [�R?Aud>�   AudP   Auf�P   [ْ�Auf��   Auf�P   Auh�P   [�e�Aui �   Auh�P   AuklP   [ِ'Auk��   AuklP   Aum�P   [ٌ�Aun�   Aum�P   AupNP   [�r�Aups�   AupNP   Aur�P   [�Aur��   Aur�P   Auu0P   [�h�AuuU�   Auu0P   Auw�P   [�kJAuw��   Auw�P   AuzP   [��Auz7�   AuzP   Au|�P   [ق�Au|��   Au|�P   Au~�P   [٭/Au�   Au~�P   Au�eP   [�X�Au���   Au�eP   Au��P   [ٵ�Au���   Au��P   Au�GP   [�?�Au�l�   Au�GP   Au��P   [ٯBAu���   Au��P   Au�)P   [�OAu�N�   Au�)P   Au��P   [م+Au���   Au��P   Au�P   [ٹ�Au�0�   Au�P   Au�|P   [��Au���   Au�|P   Au��P   [�y%Au��   Au��P   Au�^P   [��Au���   Au�^P   Au��P   [�t4Au���   Au��P   Au�@P   [�z�Au�e�   Au�@P   Au��P   [���Au���   Au��P   Au�"P   [ٮ-Au�G�   Au�"P   Au��P   [�zAu���   Au��P   Au�P   [ل�Au�)�   Au�P   Au�uP   [�I�Au���   Au�uP   Au��P   [�SKAu��   Au��P   Au�WP   [ٹ�Au�|�   Au�WP   Au��P   [�2�Au���   Au��P   Au�9P   [�O?Au�^�   Au�9P   Au��P   [�)�Au���   Au��P   Au�P   [�d�Au�@�   Au�P   Au��P   [٨�Au���   Au��P   Au��P   [َ�Au�"�   Au��P   Au�nP   [٩�Au���   Au�nP   Au��P   [�kAu��   Au��P   Au�PP   [�.�Au�u�   Au�PP   Au��P   [ڡ�Au���   Au��P   Au�2P   [�wVAu�W�   Au�2P   AuʣP   [��Au���   AuʣP   Au�P   [�(�Au�9�   Au�P   AuυP   [�@NAuϪ�   AuυP   Au��P   [� rAu��   Au��P   Au�gP   [��?AuԌ�   Au�gP   Au��P   [�D=Au���   Au��P   Au�IP   [�8Au�n�   Au�IP   AuۺP   [��cAu���   AuۺP   Au�+P   [�8Au�P�   Au�+P   Au��P   [�A�Au���   Au��P   Au�P   [�R�Au�2�   Au�P   Au�~P   [��$Au��   Au�~P   Au��P   [��AAu��   Au��P   Au�`P   [��@Au��   Au�`P   Au��P   [��Au���   Au��P   Au�BP   [��lAu�g�   Au�BP   Au�P   [ڄ8Au���   Au�P   Au�$P   [��Au�I�   Au�$P   Au��P   [��Au���   Au��P   Au�P   [�P)Au�+�   Au�P   Au�wP   [�vAu���   Au�wP   Au��P   [��Au��   Au��P   Av YP   [�V�Av ~�   Av YP   Av�P   [�E�Av��   Av�P   Av;P   [�JFAv`�   Av;P   Av�P   [قAv��   Av�P   Av
P   [ٳ�Av
B�   Av
P   Av�P   [��DAv��   Av�P   Av�P   [ٜ�Av$�   Av�P   AvpP   [�	YAv��   AvpP   Av�P   [ٲ�Av�   Av�P   AvRP   [�t�Avw�   AvRP   Av�P   [�K�Av��   Av�P   Av4P   [ٹ�AvY�   Av4P   Av�P   [�QAv��   Av�P   Av P   [�d�Av ;�   Av P   Av"�P   [�&�Av"��   Av"�P   Av$�P   [�qtAv%�   Av$�P   Av'iP   [���Av'��   Av'iP   Av)�P   [�<�Av)��   Av)�P   Av,KP   [�VdAv,p�   Av,KP   Av.�P   [���Av.��   Av.�P   Av1-P   [��Av1R�   Av1-P   Av3�P   [�!�Av3��   Av3�P   Av6P   [ْ	Av64�   Av6P   Av8�P   [��|Av8��   Av8�P   Av:�P   [�L�Av;�   Av:�P   Av=bP   [�_�Av=��   Av=bP   Av?�P   [ٽ�Av?��   Av?�P   AvBDP   [ٻ&AvBi�   AvBDP   AvD�P   [�+JAvD��   AvD�P   AvG&P   [�u�AvGK�   AvG&P   AvI�P   [ٍ�AvI��   AvI�P   AvLP   [�̗AvL-�   AvLP   AvNyP   [ّ&AvN��   AvNyP   AvP�P   [�$�AvQ�   AvP�P   AvS[P   [ٷkAvS��   AvS[P   AvU�P   [��AvU��   AvU�P   AvX=P   [ً�AvXb�   AvX=P   AvZ�P   [�$�AvZ��   AvZ�P   Av]P   [�.rAv]D�   Av]P   Av_�P   [�o@Av_��   Av_�P   AvbP   [ٯ{Avb&�   AvbP   AvdrP   [���Avd��   AvdrP   Avf�P   [�LAvg�   Avf�P   AviTP   [�F�