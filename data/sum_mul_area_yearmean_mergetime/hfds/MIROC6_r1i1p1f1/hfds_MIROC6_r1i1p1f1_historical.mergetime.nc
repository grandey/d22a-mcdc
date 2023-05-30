CDF   �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       0MIROC6 (2017): 
aerosol: SPRINTARS6.0
atmos: CCSR AGCM (T85; 256 x 128 longitude/latitude; 81 levels; top level 0.004 hPa)
atmosChem: none
land: MATSIRO6.0
landIce: none
ocean: COCO4.9 (tripolar primarily 1deg; 360 x 256 longitude/latitude; 63 levels; top grid cell 0-2 m)
ocnBgchem: none
seaIce: COCO4.9   institution      QJAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2021-01-18T07:27:07Z   data_specs_version        01.00.31   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Jhttps://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.historical.none.r1i1p1f1     grid      -native ocean tripolar grid with 360x256 cells      
grid_label        gn     history      �Tue May 30 16:59:12 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.hfds.gn.v20210129/hfds_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.yearmean.mul.areacello_historical_v20190311.fldsum.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.hfds.gn.v20210129/hfds_Omon_MIROC6_historical_r1i1p1f1_gn_195001-201412.yearmean.mul.areacello_historical_v20190311.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/MIROC6_r1i1p1f1/hfds_MIROC6_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 00:31:42 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.hfds.gn.v20210129/hfds_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc /Users/benjamin/Data/p22b/CMIP6/areacello/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Ofx.areacello.gn.v20190311/areacello_Ofx_MIROC6_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Omon.hfds.gn.v20210129/hfds_Omon_MIROC6_historical_r1i1p1f1_gn_185001-194912.yearmean.mul.areacello_historical_v20190311.fldsum.nc
2021-01-18T07:27:07Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        MIROC      mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      MIROC6     parent_time_units         days since 3200-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      	source_id         MIROC6     source_type       	AOGCM AER      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ACreation Date:(22 July 2019) MD5:b4cefb4b6dbb146fea9677a552a00934      title          MIROC6 output prepared for CMIP6   variable_id       hfds   variant_label         r1i1p1f1   license      !CMIP6 model data produced by MIROC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      cmor_version      3.5.0      tracking_id       1hdl:21.14100/01130be3-ba07-4f68-af7c-8ae87d8342bb      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T                  	time_bnds                                    lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y                   hfds                      standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    original_name         HFLX   original_units        W/m^2      history       �2021-01-18T07:27:07Z altered by CMOR: Converted units from 'W/m^2' to 'W m-2'. 2021-01-18T07:27:07Z altered by CMOR: replaced missing value flag (-999) and corresponding data with standard missing value (1e+20).    cell_measures         area: areacello              eacello                              Aq���   Aq��P   Aq�P   U�h�Aq�6�   Aq�P   Aq��P   W�4�Aq���   Aq��P   Aq��P   V�X�Aq��   Aq��P   Aq�dP   W�f�Aq���   Aq�dP   Aq��P   V�Aq���   Aq��P   Aq�FP   ����Aq�k�   Aq�FP   Aq��P   WO6yAq���   Aq��P   Aq�(P   W� aAq�M�   Aq�(P   Aq��P   W-cAq���   Aq��P   Aq�
P   Vި�Aq�/�   Aq�
P   Aq�{P   VZ5Aq���   Aq�{P   Aq��P   V*T�Aq��   Aq��P   Aq�]P   ֺ�AqĂ�   Aq�]P   Aq��P   ׬3QAq���   Aq��P   Aq�?P   W�P�Aq�d�   Aq�?P   Aq˰P   W���Aq���   Aq˰P   Aq�!P   W���Aq�F�   Aq�!P   AqВP   W�kAqз�   AqВP   Aq�P   W��Aq�(�   Aq�P   Aq�tP   ���Aqՙ�   Aq�tP   Aq��P   Wo��Aq�
�   Aq��P   Aq�VP   WCAq�{�   Aq�VP   Aq��P   Wu�Aq���   Aq��P   Aq�8P   �[s�Aq�]�   Aq�8P   Aq�P   �N�Aq���   Aq�P   Aq�P   WT�yAq�?�   Aq�P   Aq�P   V��Aq��   Aq�P   Aq��P   W�Y�Aq�!�   Aq��P   Aq�mP   W���Aq��   Aq�mP   Aq��P   WNAq��   Aq��P   Aq�OP   V�'�Aq�t�   Aq�OP   Aq��P   V���Aq���   Aq��P   Aq�1P   W[�Aq�V�   Aq�1P   Aq��P   ��Aq���   Aq��P   Aq�P   ׮A�Aq�8�   Aq�P   Aq��P   V��YAq���   Aq��P   Aq��P   W��Aq��   Aq��P   ArfP   W=��Ar��   ArfP   Ar�P   Vh�Ar��   Ar�P   ArHP   W�_�Arm�   ArHP   Ar�P   Wp'�Ar��   Ar�P   Ar*P   �Ɵ�ArO�   Ar*P   Ar�P   W�Ar��   Ar�P   ArP   W�rAr1�   ArP   Ar}P   V���Ar��   Ar}P   Ar�P   WV�Ar�   Ar�P   Ar_P   W�8Ar��   Ar_P   Ar�P   V�%�Ar��   Ar�P   ArAP   Va�NArf�   ArAP   Ar�P   Wd�VAr��   Ar�P   Ar!#P   V0��Ar!H�   Ar!#P   Ar#�P   U�9RAr#��   Ar#�P   Ar&P   VU�Ar&*�   Ar&P   Ar(vP   ֟�Ar(��   Ar(vP   Ar*�P   W<�2Ar+�   Ar*�P   Ar-XP   W��Ar-}�   Ar-XP   Ar/�P   W;��Ar/��   Ar/�P   Ar2:P   W5�Ar2_�   Ar2:P   Ar4�P   W���Ar4��   Ar4�P   Ar7P   ՛��Ar7A�   Ar7P   Ar9�P   W*ZAr9��   Ar9�P   Ar;�P   W���Ar<#�   Ar;�P   Ar>oP   � �Ar>��   Ar>oP   Ar@�P   ����ArA�   Ar@�P   ArCQP   W?ArCv�   ArCQP   ArE�P   W3��ArE��   ArE�P   ArH3P   W��eArHX�   ArH3P   ArJ�P   V��ArJ��   ArJ�P   ArMP   �p2_ArM:�   ArMP   ArO�P   W�ArO��   ArO�P   ArQ�P   W�ŉArR�   ArQ�P   ArThP   W��0ArT��   ArThP   ArV�P   U���ArV��   ArV�P   ArYJP   V��7ArYo�   ArYJP   Ar[�P   V�SAr[��   Ar[�P   Ar^,P   VX)Ar^Q�   Ar^,P   Ar`�P   VŨ�Ar`��   Ar`�P   ArcP   W^��Arc3�   ArcP   AreP   W��Are��   AreP   Arg�P   W OWArh�   Arg�P   ArjaP   W�?JArj��   ArjaP   Arl�P   Wg��Arl��   Arl�P   AroCP   V�˙Aroh�   AroCP   Arq�P   Wy�3Arq��   Arq�P   Art%P   Wb��ArtJ�   Art%P   Arv�P   W��dArv��   Arv�P   AryP   W��^Ary,�   AryP   Ar{xP   V�_Ar{��   Ar{xP   Ar}�P   V��Ar~�   Ar}�P   Ar�ZP   W��mAr��   Ar�ZP   Ar��P   W���Ar���   Ar��P   Ar�<P   W��Ar�a�   Ar�<P   Ar��P   W@�QAr���   Ar��P   Ar�P   U���Ar�C�   Ar�P   Ar��P   ֐�Ar���   Ar��P   Ar� P   �`ςAr�%�   Ar� P   Ar�qP   W���Ar���   Ar�qP   Ar��P   X=
Ar��   Ar��P   Ar�SP   W(�Ar�x�   Ar�SP   Ar��P   W�15Ar���   Ar��P   Ar�5P   W,�cAr�Z�   Ar�5P   Ar��P   Wx�Ar���   Ar��P   Ar�P   W�Ar�<�   Ar�P   Ar��P   WP2�Ar���   Ar��P   Ar��P   W�qAr��   Ar��P   Ar�jP   W��BAr���   Ar�jP   Ar��P   Uk	�Ar� �   Ar��P   Ar�LP   �u�Ar�q�   Ar�LP   Ar��P   W)nAr���   Ar��P   Ar�.P   W�RAr�S�   Ar�.P   Ar��P   W���Ar���   Ar��P   Ar�P   W��Ar�5�   Ar�P   Ar��P   �&#�Ar���   Ar��P   Ar��P   �3լAr��   Ar��P   Ar�cP   V�ǱAr���   Ar�cP   Ar��P   WW�Ar���   Ar��P   Ar�EP   W���Ar�j�   Ar�EP   ArĶP   V��vAr���   ArĶP   Ar�'P   ՟��Ar�L�   Ar�'P   ArɘP   W�;bArɽ�   ArɘP   Ar�	P   W�2;Ar�.�   Ar�	P   Ar�zP   W1*�ArΟ�   Ar�zP   Ar��P   W�Y'Ar��   Ar��P   Ar�\P   W�9"ArӁ�   Ar�\P   Ar��P   Uq.hAr���   Ar��P   Ar�>P   �t��Ar�c�   Ar�>P   ArگP   W��Ar���   ArگP   Ar� P   W�ЬAr�E�   Ar� P   ArߑP   V���Ar߶�   ArߑP   Ar�P   W5Y�Ar�'�   Ar�P   Ar�sP   W�'TAr��   Ar�sP   Ar��P   X��Ar�	�   Ar��P   Ar�UP   V�)�Ar�z�   Ar�UP   Ar��P   ��z�Ar���   Ar��P   Ar�7P   ���Ar�\�   Ar�7P   Ar�P   WP�_Ar���   Ar�P   Ar�P   W�>zAr�>�   Ar�P   Ar��P   W�J�Ar���   Ar��P   Ar��P   Ws��Ar� �   Ar��P   Ar�lP   W� gAr���   Ar�lP   Ar��P   W��Ar��   Ar��P   Ar�NP   Նc�Ar�s�   Ar�NP   As�P   �Cs�As��   As�P   As0P   ց��AsU�   As0P   As�P   W.�As��   As�P   As	P   W��MAs	7�   As	P   As�P   XFs�As��   As�P   As�P   W�.�As�   As�P   AseP   W� �As��   AseP   As�P   WL�NAs��   As�P   AsGP   WA��Asl�   AsGP   As�P   W�As��   As�P   As)P   W�xAsN�   As)P   As�P   W���As��   As�P   AsP   V!^fAs0�   AsP   As!|P   W��9As!��   As!|P   As#�P   X�As$�   As#�P   As&^P   W�k�As&��   As&^P   As(�P   WbJ�As(��   As(�P   As+@P   V���As+e�   As+@P   As-�P   X��As-��   As-�P   As0"P   X��As0G�   As0"P   As2�P   X �|As2��   As2�P   As5P   V��{As5)�   As5P   As7uP   ���