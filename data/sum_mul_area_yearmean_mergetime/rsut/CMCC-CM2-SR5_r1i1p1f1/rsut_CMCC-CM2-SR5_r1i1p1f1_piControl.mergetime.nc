CDF  �   
      time       bnds      lon       lat          2   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �CMCC-CM2-SR5 (2016): 
aerosol: MAM3
atmos: CAM5.3 (1deg; 288 x 192 longitude/latitude; 30 levels; top at ~2 hPa)
atmosChem: none
land: CLM4.5 (BGC mode)
landIce: none
ocean: NEMO3.6 (ORCA1 tripolar primarly 1 deg lat/lon with meridional refinement down to 1/3 degree in the tropics; 362 x 292 longitude/latitude; 50 vertical levels; top grid cell 0-1 m)
ocnBgchem: none
seaIce: CICE4.0      institution       QFondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    comment       none   contact       	T. Lovato      creation_date         2020-06-10T15:45:10Z   data_specs_version        01.00.31   
experiment        pre-industrial control     experiment_id         	piControl      external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.CMCC.CMCC-CM2-SR5.piControl.none.r1i1p1f1     grid      native atmosphere regular grid     
grid_label        gn     history      5Tue May 30 16:58:33 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.piControl.r1i1p1f1.Amon.rsut.gn.v20200616/rsut_Amon_CMCC-CM2-SR5_piControl_r1i1p1f1_gn_185001-209912.yearmean.mul.areacella_piControl_v20200616.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.piControl.r1i1p1f1.Amon.rsut.gn.v20200616/rsut_Amon_CMCC-CM2-SR5_piControl_r1i1p1f1_gn_210001-234912.yearmean.mul.areacella_piControl_v20200616.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsut/CMCC-CM2-SR5_r1i1p1f1/rsut_CMCC-CM2-SR5_r1i1p1f1_piControl.mergetime.nc
Fri Nov 04 06:21:02 2022: cdo -O -s -fldsum -setattribute,rsut@units=W m-2 m2 -mul -yearmean -selname,rsut /Users/benjamin/Data/p22b/CMIP6/rsut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.piControl.r1i1p1f1.Amon.rsut.gn.v20200616/rsut_Amon_CMCC-CM2-SR5_piControl_r1i1p1f1_gn_185001-209912.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.piControl.r1i1p1f1.fx.areacella.gn.v20200616/areacella_fx_CMCC-CM2-SR5_piControl_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.piControl.r1i1p1f1.Amon.rsut.gn.v20200616/rsut_Amon_CMCC-CM2-SR5_piControl_r1i1p1f1_gn_185001-209912.yearmean.mul.areacella_piControl_v20200616.fldsum.nc
2020-06-10T15:45:10Z ;rewrote data to be consistent with CMIP for variable rsut found in table Amon.;
none      initialization_index            institution_id        CMCC   mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      CMCC-CM2-SR5   parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      
references        none   run_variant       1st realization    	source_id         CMCC-CM2-SR5   source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(05 February 2020) MD5:6a248fd76c55aa6d6f7b3cc6866b5faf      title         &CMCC-CM2-SR5 output prepared for CMIP6     variable_id       rsut   variant_label         r1i1p1f1   license      ?CMIP6 model data produced by CMCC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.5.0      tracking_id       1hdl:21.14100/27530f58-d11c-4e65-a25f-69182fc402b7      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsut                   	   standard_name         toa_outgoing_shortwave_flux    	long_name          TOA Outgoing Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       at the top of the atmosphere   original_name         FSUTOA     cell_measures         area: areacella             �eacella             �                Aq���   Aq��P   Aq�P   [0R�Aq�6�   Aq�P   Aq��P   [0��Aq���   Aq��P   Aq��P   [0O�Aq��   Aq��P   Aq�dP   [06~Aq���   Aq�dP   Aq��P   [0PAq���   Aq��P   Aq�FP   [/�Aq�k�   Aq�FP   Aq��P   [0*�Aq���   Aq��P   Aq�(P   [0�[Aq�M�   Aq�(P   Aq��P   [0�Aq���   Aq��P   Aq�
P   [0 Aq�/�   Aq�
P   Aq�{P   [/�uAq���   Aq�{P   Aq��P   [0A�Aq��   Aq��P   Aq�]P   [0ZAqĂ�   Aq�]P   Aq��P   [/�)Aq���   Aq��P   Aq�?P   [0�Aq�d�   Aq�?P   Aq˰P   [0��Aq���   Aq˰P   Aq�!P   [0vAq�F�   Aq�!P   AqВP   [0d�Aqз�   AqВP   Aq�P   [0_�Aq�(�   Aq�P   Aq�tP   [0o�Aqՙ�   Aq�tP   Aq��P   [0��Aq�
�   Aq��P   Aq�VP   [0�kAq�{�   Aq�VP   Aq��P   [03�Aq���   Aq��P   Aq�8P   [0�MAq�]�   Aq�8P   Aq�P   [0AAq���   Aq�P   Aq�P   [0L�Aq�?�   Aq�P   Aq�P   [0�Aq��   Aq�P   Aq��P   [0j!Aq�!�   Aq��P   Aq�mP   [0��Aq��   Aq�mP   Aq��P   [0�Aq��   Aq��P   Aq�OP   [1Aq�t�   Aq�OP   Aq��P   [0עAq���   Aq��P   Aq�1P   [0�tAq�V�   Aq�1P   Aq��P   [0�%Aq���   Aq��P   Aq�P   [0ϡAq�8�   Aq�P   Aq��P   [1W�Aq���   Aq��P   Aq��P   [0I�Aq��   Aq��P   ArfP   [0o.Ar��   ArfP   Ar�P   [1PAr��   Ar�P   ArHP   [0�Arm�   ArHP   Ar�P   [0��Ar��   Ar�P   Ar*P   [01#ArO�   Ar*P   Ar�P   [1c�Ar��   Ar�P   ArP   [0�xAr1�   ArP   Ar}P   [1O�Ar��   Ar}P   Ar�P   [0��Ar�   Ar�P   Ar_P   [1p�Ar��   Ar_P   Ar�P   [0�Ar��   Ar�P   ArAP   [0�<Arf�   ArAP   Ar�P   [1)�Ar��   Ar�P   Ar!#P   [1��Ar!H�   Ar!#P   Ar#�P   [0�rAr#��   Ar#�P   Ar&P   [0��Ar&*�   Ar&P   Ar(vP   [0�AAr(��   Ar(vP   Ar*�P   [1K�Ar+�   Ar*�P   Ar-XP   [1�vAr-}�   Ar-XP   Ar/�P   [1I�Ar/��   Ar/�P   Ar2:P   [1sAr2_�   Ar2:P   Ar4�P   [1 �Ar4��   Ar4�P   Ar7P   [0��Ar7A�   Ar7P   Ar9�P   [1_Ar9��   Ar9�P   Ar;�P   [0�PAr<#�   Ar;�P   Ar>oP   [0�cAr>��   Ar>oP   Ar@�P   [0�"ArA�   Ar@�P   ArCQP   [0��ArCv�   ArCQP   ArE�P   [1�ArE��   ArE�P   ArH3P   [0��ArHX�   ArH3P   ArJ�P   [1.�ArJ��   ArJ�P   ArMP   [1.YArM:�   ArMP   ArO�P   [0�	ArO��   ArO�P   ArQ�P   [0�ArR�   ArQ�P   ArThP   [0��ArT��   ArThP   ArV�P   [1	BArV��   ArV�P   ArYJP   [1�ArYo�   ArYJP   Ar[�P   [0��Ar[��   Ar[�P   Ar^,P   [0�UAr^Q�   Ar^,P   Ar`�P   [0�Ar`��   Ar`�P   ArcP   [0�Arc3�   ArcP   AreP   [07Are��   AreP   Arg�P   [/�KArh�   Arg�P   ArjaP   [/ǦArj��   ArjaP   Arl�P   [0D�Arl��   Arl�P   AroCP   [0Aroh�   AroCP   Arq�P   [1�Arq��   Arq�P   Art%P   [1;�ArtJ�   Art%P   Arv�P   [0=�Arv��   Arv�P   AryP   [1hBAry,�   AryP   Ar{xP   [0��Ar{��   Ar{xP   Ar}�P   [0�?Ar~�   Ar}�P   Ar�ZP   [0�mAr��   Ar�ZP   Ar��P   [0fAr���   Ar��P   Ar�<P   [0[Ar�a�   Ar�<P   Ar��P   [1�Ar���   Ar��P   Ar�P   [1�Ar�C�   Ar�P   Ar��P   [0�Ar���   Ar��P   Ar� P   [0��Ar�%�   Ar� P   Ar�qP   [0�-Ar���   Ar�qP   Ar��P   [1-Ar��   Ar��P   Ar�SP   [0+Ar�x�   Ar�SP   Ar��P   [0�kAr���   Ar��P   Ar�5P   [1N9Ar�Z�   Ar�5P   Ar��P   [0��Ar���   Ar��P   Ar�P   [0ڪAr�<�   Ar�P   Ar��P   [/��Ar���   Ar��P   Ar��P   [0��Ar��   Ar��P   Ar�jP   [0h$Ar���   Ar�jP   Ar��P   [0��Ar� �   Ar��P   Ar�LP   [0`�Ar�q�   Ar�LP   Ar��P   [0�Ar���   Ar��P   Ar�.P   [0�@Ar�S�   Ar�.P   Ar��P   [/НAr���   Ar��P   Ar�P   [0y�Ar�5�   Ar�P   Ar��P   [0�Ar���   Ar��P   Ar��P   [0j�Ar��   Ar��P   Ar�cP   [0NAr���   Ar�cP   Ar��P   [/��Ar���   Ar��P   Ar�EP   [0roAr�j�   Ar�EP   ArĶP   [0vjAr���   ArĶP   Ar�'P   [0�Ar�L�   Ar�'P   ArɘP   [0hMArɽ�   ArɘP   Ar�	P   [0 �Ar�.�   Ar�	P   Ar�zP   [0[ArΟ�   Ar�zP   Ar��P   [0�iAr��   Ar��P   Ar�\P   [0��ArӁ�   Ar�\P   Ar��P   [/{�Ar���   Ar��P   Ar�>P   [0�aAr�c�   Ar�>P   ArگP   [/��Ar���   ArگP   Ar� P   [/��Ar�E�   Ar� P   ArߑP   [/1Ar߶�   ArߑP   Ar�P   [/�1Ar�'�   Ar�P   Ar�sP   [0��Ar��   Ar�sP   Ar��P   [0kAr�	�   Ar��P   Ar�UP   [0yAr�z�   Ar�UP   Ar��P   [0'�Ar���   Ar��P   Ar�7P   [/�Ar�\�   Ar�7P   Ar�P   [/��Ar���   Ar�P   Ar�P   [0{�Ar�>�   Ar�P   Ar��P   [0��Ar���   Ar��P   Ar��P   [0?�Ar� �   Ar��P   Ar�lP   [0OAr���   Ar�lP   Ar��P   [0ESAr��   Ar��P   Ar�NP   [0y3Ar�s�   Ar�NP   As�P   [0�/As��   As�P   As0P   [0H AsU�   As0P   As�P   [0��As��   As�P   As	P   [0�As	7�   As	P   As�P   [0As��   As�P   As�P   [/��As�   As�P   AseP   [0;As��   AseP   As�P   [/�aAs��   As�P   AsGP   [/Q0Asl�   AsGP   As�P   [/��As��   As�P   As)P   [/��AsN�   As)P   As�P   [0�As��   As�P   AsP   [0	UAs0�   AsP   As!|P   [04�As!��   As!|P   As#�P   [/�MAs$�   As#�P   As&^P   [/�<As&��   As&^P   As(�P   [00�As(��   As(�P   As+@P   [0�As+e�   As+@P   As-�P   [/�xAs-��   As-�P   As0"P   [09As0G�   As0"P   As2�P   [0(%As2��   As2�P   As5P   [0�As5)�   As5P   As7uP   [0|�As7��   As7uP   As9�P   [0^�As:�   As9�P   As<WP   [0��As<|�   As<WP   As>�P   [1�As>��   As>�P   AsA9P   [0�AsA^�   AsA9P   AsC�P   [0��AsC��   AsC�P   AsFP   [0�AsF@�   AsFP   AsH�P   [0ͶAsH��   AsH�P   AsJ�P   [/��AsK"�   AsJ�P   AsMnP   [0hgAsM��   AsMnP   AsO�P   [0�AsP�   AsO�P   AsRPP   [0��AsRu�   AsRPP   AsT�P   [/�#AsT��   AsT�P   AsW2P   [0��AsWW�   AsW2P   AsY�P   [0��AsY��   AsY�P   As\P   [0�As\9�   As\P   As^�P   [1�As^��   As^�P   As`�P   [09Asa�   As`�P   AscgP   [0[4Asc��   AscgP   Ase�P   [0�BAse��   Ase�P   AshIP   [0DJAshn�   AshIP   Asj�P   [0�!Asj��   Asj�P   Asm+P   [0hAsmP�   Asm+P   Aso�P   [0YbAso��   Aso�P   AsrP   [/��Asr2�   AsrP   Ast~P   [/�Ast��   Ast~P   Asv�P   [0.Asw�   Asv�P   Asy`P   [0Asy��   Asy`P   As{�P   [0<�As{��   As{�P   As~BP   [0MAs~g�   As~BP   As��P   [/�6As���   As��P   As�$P   [/~RAs�I�   As�$P   As��P   [0%/As���   As��P   As�P   [/�As�+�   As�P   As�wP   [/nAs���   As�wP   As��P   [/z2As��   As��P   As�YP   [/�_As�~�   As�YP   As��P   [/�gAs���   As��P   As�;P   [/�MAs�`�   As�;P   As��P   [/�EAs���   As��P   As�P   [/��As�B�   As�P   As��P   [0bAs���   As��P   As��P   [/��As�$�   As��P   As�pP   [/�RAs���   As�pP   As��P   [0MAs��   As��P   As�RP   [/ǞAs�w�   As�RP   As��P   [0QAs���   As��P   As�4P   [/�JAs�Y�   As�4P   As��P   [0��As���   As��P   As�P   [0M�As�;�   As�P   As��P   [0�HAs���   As��P   As��P   [02iAs��   As��P   As�iP   [0gMAs���   As�iP   As��P   [0��As���   As��P   As�KP   [0�_As�p�   As�KP   As��P   [0��As���   As��P   As�-P   [0ޞAs�R�   As�-P   AsP   [17As���   AsP   As�P   [0c�As�4�   As�P   AsǀP   [0'�Asǥ�   AsǀP   As��P   [0�GAs��   As��P   As�bP   [1�Aṡ�   As�bP   As��P   [1
�As���   As��P   As�DP   [1S'As�i�   As�DP   AsӵP   [0٭As���   AsӵP   As�&P   [0��As�K�   As�&P   AsؗP   [0}lAsؼ�   AsؗP   As�P   [0ȈAs�-�   As�P   As�yP   [0�hAsݞ�   As�yP   As��P   [0�As��   As��P   As�[P   [0��As��   As�[P   As��P   [1
�As���   As��P   As�=P   [0�*As�b�   As�=P   As�P   [18As���   As�P   As�P   [0�tAs�D�   As�P   As�P   [0��As��   As�P   As�P   [0��As�&�   As�P   As�rP   [1#lAs��   As�rP   As��P   [1 bAs��   As��P   As�TP   [1 JAs�y�   As�TP   As��P   [0u�As���   As��P   As�6P   [1\�As�[�   As�6P   As��P   [13As���   As��P   AtP   [0�6At=�   AtP   At�P   [0�At��   At�P   At�P   [0�,At�   At�P   At	kP   [18oAt	��   At	kP   At�P   [1At�   At�P   AtMP   [0�QAtr�   AtMP   At�P   [0��At��   At�P   At/P   [1k�AtT�   At/P   At�P   [1AnAt��   At�P   AtP   [0�At6�   AtP   At�P   [1W�At��   At�P   At�P   [0�-At�   At�P   AtdP   [0�At��   AtdP   At!�P   [0��At!��   At!�P   At$FP   [0O4At$k�   At$FP   At&�P   [/�At&��   At&�P   At)(P   [0�At)M�   At)(P   At+�P   [09-At+��   At+�P   At.
P   [0)�At./�   At.
P   At0{P   [/��At0��   At0{P   At2�P   [/O�At3�   At2�P   At5]P   [0MPAt5��   At5]P   At7�P   [/xeAt7��   At7�P   At:?P   [/�At:d�   At:?P   At<�P   [0�At<��   At<�P   At?!P   [/��At?F�   At?!P   AtA�P   [/�bAtA��   AtA�P   AtDP   [0J0AtD(�   AtDP   AtFtP   [/�RAtF��   AtFtP   AtH�P   [0(AtI
�   AtH�P   AtKVP   [/��AtK{�   AtKVP   AtM�P   [0�AtM��   AtM�P   AtP8P   [0}�AtP]�   AtP8P   AtR�P   [/�AtR��   AtR�P   AtUP   [/�gAtU?�   AtUP   AtW�P   [/��AtW��   AtW�P   AtY�P   [0#AtZ!�   AtY�P   At\mP   [/��At\��   At\mP   At^�P   [/�4At_�   At^�P   AtaOP   [0a@Atat�   AtaOP   Atc�P   [08�Atc��   Atc�P   Atf1P   [/�xAtfV�   Atf1P   Ath�P   [/��Ath��   Ath�P   AtkP   [/PAtk8�   AtkP   Atm�P   [/�3Atm��   Atm�P   Ato�P   [0��Atp�   Ato�P   AtrfP   [0p�Atr��   AtrfP   Att�P   [/��Att��   Att�P   AtwHP   [0�Atwm�   AtwHP   Aty�P   [/��Aty��   Aty�P   At|*P   [0B�At|O�   At|*P   At~�P   [0� At~��   At~�P   At�P   [/�qAt�1�   At�P   At�}P   [0At���   At�}P   At��P   [0:dAt��   At��P   At�_P   [1GJAt���   At�_P   At��P   [0k�At���   At��P   At�AP   [0�At�f�   At�AP   At��P   [0H�At���   At��P   At�#P   [0nAt�H�   At�#P   At��P   [0s�At���   At��P   At�P   [0p�At�*�   At�P   At�vP   [0��At���   At�vP   At��P   [0��At��   At��P   At�XP   [0��At�}�   At�XP   At��P   [0�$At���   At��P   At�:P   [0�wAt�_�   At�:P   At��P   [0"At���   At��P   At�P   [0�At�A�   At�P   At��P   [/�ZAt���   At��P   At��P   [0Z�At�#�   At��P   At�oP   [0;�At���   At�oP   At��P   [0�(At��   At��P   At�QP   [0�At�v�   At�QP   At��P   [0�MAt���   At��P   At�3P   [0�XAt�X�   At�3P   At��P   [0}�At���   At��P   At�P   [0�At�:�   At�P   At��P   [0�sAt���   At��P   At��P   [1�HAt��   At��P   At�hP   [0.�Atō�   At�hP   At��P   [0�At���   At��P   At�JP   [0�'At�o�   At�JP   At̻P   [0��At���   At̻P   At�,P   [0��At�Q�   At�,P   AtѝP   [0[�At���   AtѝP   At�P   [0�At�3�   At�P   At�P   [0�vAt֤�   At�P   At��P   [0S�At��   At��P   At�aP   [/��Atۆ�   At�aP   At��P   [/��At���   At��P   At�CP   [0*�At�h�   At�CP   At�P   [0l7At���   At�P   At�%P   [/��At�J�   At�%P   At�P   [/WAt��   At�P   At�P   [/��At�,�   At�P   At�xP   [/n�At��   At�xP   At��P   [/��At��   At��P   At�ZP   [/�At��   At�ZP   At��P   [/CUAt���   At��P   At�<P   [/�{At�a�   At�<P   At��P   [/�9At���   At��P   At�P   [/��At�C�   At�P   At��P   [/b�At���   At��P   Au  P   [/+�Au %�   Au  P   AuqP   [.�Au��   AuqP   Au�P   [/��Au�   Au�P   AuSP   [/I_Aux�   AuSP   Au	�P   [.̤Au	��   Au	�P   Au5P   [/mSAuZ�   Au5P   Au�P   [/�Au��   Au�P   AuP   [/=TAu<�   AuP   Au�P   [/�^Au��   Au�P   Au�P   [.�wAu�   Au�P   AujP   [0�/Au��   AujP   Au�P   [/�Au �   Au�P   AuLP   [/�AAuq�   AuLP   Au�P   [/�Au��   Au�P   Au".P   [/"�Au"S�   Au".P   Au$�P   [/�
Au$��   Au$�P   Au'P   [0�FAu'5�   Au'P   Au)�P   [/��Au)��   Au)�P   Au+�P   [/�mAu,�   Au+�P   Au.cP   [/��Au.��   Au.cP   Au0�P   [/�KAu0��   Au0�P   Au3EP   [0�Au3j�   Au3EP   Au5�P   [1=Au5��   Au5�P   Au8'P   [0L#Au8L�   Au8'P   Au:�P   [0�Au:��   Au:�P   Au=	P   [/�Au=.�   Au=	P   Au?zP   [0�{Au?��   Au?zP   AuA�P   [0��AuB�   AuA�P   AuD\P   [0b�AuD��   AuD\P   AuF�P   [0�TAuF��   AuF�P   AuI>P   [0�+AuIc�   AuI>P   AuK�P   [0ۆAuK��   AuK�P   AuN P   [0]�AuNE�   AuN P   AuP�P   [0��AuP��   AuP�P   AuSP   [0��AuS'�   AuSP   AuUsP   [0�AuU��   AuUsP   AuW�P   [0��AuX	�   AuW�P   AuZUP   [0��AuZz�   AuZUP   Au\�P   [1*Au\��   Au\�P   Au_7P   [0��Au_\�   Au_7P   Aua�P   [0��Aua��   Aua�P   AudP   [0tTAud>�   AudP   Auf�P   [0��Auf��   Auf�P   Auh�P   [0	XAui �   Auh�P   AuklP   [0�JAuk��   AuklP   Aum�P   [0iqAun�   Aum�P   AupNP   [0X�Aups�   AupNP   Aur�P   [0+�Aur��   Aur�P   Auu0P   [0=�AuuU�   Auu0P   Auw�P   [0�&Auw��   Auw�P   AuzP   [0��Auz7�   AuzP   Au|�P   [0	Au|��   Au|�P   Au~�P   [0H7Au�   Au~�P   Au�eP   [0��Au���   Au�eP   Au��P   [/��Au���   Au��P   Au�GP   [0!aAu�l�   Au�GP   Au��P   [0h1Au���   Au��P   Au�)P   [1 dAu�N�   Au�)P   Au��P   [/�
Au���   Au��P   Au�P   [/��Au�0�   Au�P   Au�|P   [/��Au���   Au�|P   Au��P   [/�Au��   Au��P   Au�^P   [0�Au���   Au�^P   Au��P   [/I4Au���   Au��P   Au�@P   [/�WAu�e�   Au�@P   Au��P   [/}�Au���   Au��P   Au�"P   [/'�Au�G�   Au�"P   Au��P   [/h�Au���   Au��P   Au�P   [/�:Au�)�   Au�P   Au�uP   [/eaAu���   Au�uP   Au��P   [/4�Au��   Au��P   Au�WP   [.��Au�|�   Au�WP   Au��P   [.�Au���   Au��P   Au�9P   [.�JAu�^�   Au�9P   Au��P   [.��Au���   Au��P   Au�P   [.��Au�@�   Au�P   Au��P   [/�*Au���   Au��P   Au��P   [/$Au�"�   Au��P   Au�nP   [.��Au���   Au�nP   Au��P   [0c6Au��   Au��P   Au�PP   [/��Au�u�   Au�PP   Au��P   [.�_Au���   Au��P   Au�2P   [/�MAu�W�   Au�2P   AuʣP   [/�IAu���   AuʣP   Au�P   [/�aAu�9�   Au�P   AuυP   [0�"AuϪ�   AuυP   Au��P   [/��Au��   Au��P   Au�gP   [/�fAuԌ�   Au�gP   Au��P   [/��Au���   Au��P   Au�IP   [/�kAu�n�   Au�IP   AuۺP   [/��Au���   AuۺP   Au�+P   [/�Au�P�   Au�+P   Au��P   [0�Au���   Au��P   Au�P   [0�<Au�2�   Au�P   Au�~P   [0,�Au��   Au�~P   Au��P   [0�Au��   Au��P   Au�`P   [0��Au��   Au�`P   Au��P   [06Au���   Au��P   Au�BP   [0V�Au�g�   Au�BP   Au�P   [0CnAu���   Au�P   Au�$P   [/�Au�I�   Au�$P   Au��P   [0:�Au���   Au��P   Au�P   [0�)Au�+�   Au�P   Au�wP   [0";Au���   Au�wP   Au��P   [0jAu��   Au��P   Av YP   [0@�Av ~�   Av YP   Av�P   [0��Av��   Av�P   Av;P   [0Z�Av`�   Av;P   Av�P   [01Av��   Av�P   Av
P   [0=�Av
B�   Av
P   Av�P   [0eLAv��   Av�P   Av�P   [/�Av$�   Av�P   AvpP   [0��Av��   AvpP   Av�P   [0��Av�   Av�P   AvRP   [0UAvw�   AvRP   Av�P   [/��Av��   Av�P   Av4P   [0��AvY�   Av4P   Av�P   [1�Av��   Av�P   Av P   [0�MAv ;�   Av P   Av"�P   [13AAv"��   Av"�P   Av$�P   [0dAv%�   Av$�P   Av'iP   [0��Av'��   Av'iP   Av)�P   [1OAv)��   Av)�P   Av,KP   [0��Av,p�   Av,KP   Av.�P   [0+Av.��   Av.�P   Av1-P   [0W�Av1R�   Av1-P   Av3�P   [0�8Av3��   Av3�P   Av6P   [/�\Av64�   Av6P   Av8�P   [0DbAv8��   Av8�P   Av:�P   [05�Av;�   Av:�P   Av=bP   [//�Av=��   Av=bP   Av?�P   [/�Av?��   Av?�P   AvBDP   [/��AvBi�   AvBDP   AvD�P   [0B�AvD��   AvD�P   AvG&P   [/xPAvGK�   AvG&P   AvI�P   [/�-AvI��   AvI�P   AvLP   [0�AvL-�   AvLP   AvNyP   [0R`AvN��   AvNyP   AvP�P   [0�XAvQ�   AvP�P   AvS[P   [0:�AvS��   AvS[P   AvU�P   [/��AvU��   AvU�P   AvX=P   [.��AvXb�   AvX=P   AvZ�P   [/J�AvZ��   AvZ�P   Av]P   [/�Av]D�   Av]P   Av_�P   [/5�Av_��   Av_�P   AvbP   [/a�Avb&�   AvbP   AvdrP   [/OAvd��   AvdrP   Avf�P   [0�KAvg�   Avf�P   AviTP   [/�