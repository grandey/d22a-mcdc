CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-11-08T21:42:28Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacello      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      ;Wed Nov 09 19:01:22 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20191108.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/hfds/ACCESS-CM2_r1i1p1f1/hfds_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc
Thu Nov 03 22:04:22 2022: cdo -O -s -fldsum -setattribute,hfds@units=W m-2 m2 -mul -yearmean -selname,hfds /Users/benjamin/Data/p22b/CMIP6/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacello/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Ofx.areacello.gn.v20191108/areacello_Ofx_ACCESS-CM2_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/hfds/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Omon.hfds.gn.v20191108/hfds_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacello_historical_v20191108.fldsum.nc
2019-11-08T21:42:28Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.    initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         �Exp: CM2-historical; Local ID: bj594; Variable: hfds (['sfc_hflux_from_runoff', 'sfc_hflux_coupler', 'sfc_hflux_from_water_evap', 'sfc_hflux_from_water_prec', 'frazil_2d'])   parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0950-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       hfds   variant_label         r1i1p1f1   version       	v20191108      cmor_version      3.4.0      tracking_id       1hdl:21.14100/6f1726b3-31a7-4609-af04-cc80b98c5aab      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   hfds                   	   standard_name         'surface_downward_heat_flux_in_sea_water    	long_name         'Downward Heat Flux at Sea Water Surface    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       {This is the net flux of heat entering the liquid water column through its upper surface (excluding any 'flux adjustment') .    cell_measures         area: areacello    history       o2019-11-08T21:42:24Z altered by CMOR: replaced missing value flag (-1e+20) with standard missing value (1e+20).             �                Aq���   Aq��P   Aq�P   W	#Aq�6�   Aq�P   Aq��P   V�Y�Aq���   Aq��P   Aq��P   V�ӅAq��   Aq��P   Aq�dP   ��O�Aq���   Aq�dP   Aq��P   ׃Aq���   Aq��P   Aq�FP   �PH�Aq�k�   Aq�FP   Aq��P   V��>Aq���   Aq��P   Aq�(P   U��JAq�M�   Aq�(P   Aq��P   V��MAq���   Aq��P   Aq�
P   �_�Aq�/�   Aq�
P   Aq�{P   ̴֪Aq���   Aq�{P   Aq��P   U[�Aq��   Aq��P   Aq�]P   �@��AqĂ�   Aq�]P   Aq��P   V\��Aq���   Aq��P   Aq�?P   �	ܯAq�d�   Aq�?P   Aq˰P   V`]8Aq���   Aq˰P   Aq�!P   V�0WAq�F�   Aq�!P   AqВP   �L�!Aqз�   AqВP   Aq�P   W��jAq�(�   Aq�P   Aq�tP   W�Aqՙ�   Aq�tP   Aq��P   V�gAq�
�   Aq��P   Aq�VP   W0�$Aq�{�   Aq�VP   Aq��P   ���Aq���   Aq��P   Aq�8P   ֭t�Aq�]�   Aq�8P   Aq�P   ֳ��Aq���   Aq�P   Aq�P   V�I4Aq�?�   Aq�P   Aq�P   W:��Aq��   Aq�P   Aq��P   V�Aq�!�   Aq��P   Aq�mP   V�jAq��   Aq�mP   Aq��P   V��Aq��   Aq��P   Aq�OP   ֈ��Aq�t�   Aq�OP   Aq��P   W��1Aq���   Aq��P   Aq�1P   W#/Aq�V�   Aq�1P   Aq��P   ��` Aq���   Aq��P   Aq�P   �%c�Aq�8�   Aq�P   Aq��P   �N?Aq���   Aq��P   Aq��P   V���Aq��   Aq��P   ArfP   Wi6�Ar��   ArfP   Ar�P   ��ZAr��   Ar�P   ArHP   W'��Arm�   ArHP   Ar�P   W3Ar��   Ar�P   Ar*P   ׌�ArO�   Ar*P   Ar�P   V�Ar��   Ar�P   ArP   V���Ar1�   ArP   Ar}P   �LmAr��   Ar}P   Ar�P   W "#Ar�   Ar�P   Ar_P   W#��Ar��   Ar_P   Ar�P   ���Ar��   Ar�P   ArAP   ���2Arf�   ArAP   Ar�P   V��Ar��   Ar�P   Ar!#P   V�"Ar!H�   Ar!#P   Ar#�P   UT�=Ar#��   Ar#�P   Ar&P   V��Ar&*�   Ar&P   Ar(vP   ש	�Ar(��   Ar(vP   Ar*�P   ���Ar+�   Ar*�P   Ar-XP   URbAr-}�   Ar-XP   Ar/�P   V�-Ar/��   Ar/�P   Ar2:P   V��sAr2_�   Ar2:P   Ar4�P   �-�Ar4��   Ar4�P   Ar7P   VS��Ar7A�   Ar7P   Ar9�P   TCݚAr9��   Ar9�P   Ar;�P   ��Y9Ar<#�   Ar;�P   Ar>oP   �e ,Ar>��   Ar>oP   Ar@�P   �1�+ArA�   Ar@�P   ArCQP   U1	�ArCv�   ArCQP   ArE�P   VQѓArE��   ArE�P   ArH3P   �8�@ArHX�   ArH3P   ArJ�P   Wr�.ArJ��   ArJ�P   ArMP   V��FArM:�   ArMP   ArO�P   S�MwArO��   ArO�P   ArQ�P   V��ArR�   ArQ�P   ArThP   V�R|ArT��   ArThP   ArV�P   W#VArV��   ArV�P   ArYJP   V��ArYo�   ArYJP   Ar[�P   V�
gAr[��   Ar[�P   Ar^,P   V�-3Ar^Q�   Ar^,P   Ar`�P   W&��Ar`��   Ar`�P   ArcP   V�͘Arc3�   ArcP   AreP   W�	)Are��   AreP   Arg�P   �3�>Arh�   Arg�P   ArjaP   V�X�Arj��   ArjaP   Arl�P   Wk�zArl��   Arl�P   AroCP   W>�BAroh�   AroCP   Arq�P   WO�<Arq��   Arq�P   Art%P   U�RArtJ�   Art%P   Arv�P   V�laArv��   Arv�P   AryP   ���Ary,�   AryP   Ar{xP   W/�bAr{��   Ar{xP   Ar}�P   W���Ar~�   Ar}�P   Ar�ZP   V�Z?Ar��   Ar�ZP   Ar��P   V���Ar���   Ar��P   Ar�<P   V�S�Ar�a�   Ar�<P   Ar��P   ׁ��Ar���   Ar��P   Ar�P   V�(Ar�C�   Ar�P   Ar��P   �{}�Ar���   Ar��P   Ar� P   V�i�Ar�%�   Ar� P   Ar�qP   ք{ZAr���   Ar�qP   Ar��P   W|>wAr��   Ar��P   Ar�SP   U��Ar�x�   Ar�SP   Ar��P   �m? Ar���   Ar��P   Ar�5P   WEibAr�Z�   Ar�5P   Ar��P   �I�mAr���   Ar��P   Ar�P   V�@{Ar�<�   Ar�P   Ar��P   W4�Ar���   Ar��P   Ar��P   ���Ar��   Ar��P   Ar�jP   ְ��Ar���   Ar�jP   Ar��P   W,�XAr� �   Ar��P   Ar�LP   V�0Ar�q�   Ar�LP   Ar��P   V'�KAr���   Ar��P   Ar�.P   U��Ar�S�   Ar�.P   Ar��P   �R�Ar���   Ar��P   Ar�P   ��1+Ar�5�   Ar�P   Ar��P   �,?RAr���   Ar��P   Ar��P   �	hAr��   Ar��P   Ar�cP   �C�rAr���   Ar�cP   Ar��P   ���Ar���   Ar��P   Ar�EP   WZ"Ar�j�   Ar�EP   ArĶP   V[Z�Ar���   ArĶP   Ar�'P   ո�'Ar�L�   Ar�'P   ArɘP   փ�Arɽ�   ArɘP   Ar�	P   ���Ar�.�   Ar�	P   Ar�zP   �K�ArΟ�   Ar�zP   Ar��P   WAr��   Ar��P   Ar�\P   �p�%ArӁ�   Ar�\P   Ar��P   V�E�Ar���   Ar��P   Ar�>P   ��xAr�c�   Ar�>P   ArگP   U��#Ar���   ArگP   Ar� P   V���Ar�E�   Ar� P   ArߑP   WoQ�Ar߶�   ArߑP   Ar�P   U�?�Ar�'�   Ar�P   Ar�sP   V�s�Ar��   Ar�sP   Ar��P   V�AAr�	�   Ar��P   Ar�UP   V�y~Ar�z�   Ar�UP   Ar��P   Vm�Ar���   Ar��P   Ar�7P   ֝s�Ar�\�   Ar�7P   Ar�P   V��Ar���   Ar�P   Ar�P   ֿs6Ar�>�   Ar�P   Ar��P   W��Ar���   Ar��P   Ar��P   Wp�aAr� �   Ar��P   Ar�lP   WҔ�Ar���   Ar�lP   Ar��P   Wc�Ar��   Ar��P   Ar�NP   ׼k:Ar�s�   Ar�NP   As�P   ץT�As��   As�P   As0P   �)V<AsU�   As0P   As�P   W&��As��   As�P   As	P   W��wAs	7�   As	P   As�P   W���As��   As�P   As�P   W[��As�   As�P   AseP   W�P�As��   AseP   As�P   WtF�As��   As�P   AsGP   W�O�Asl�   AsGP   As�P   W�WWAs��   As�P   As)P   W���AsN�   As)P   As�P   W�pNAs��   As�P   AsP   W���As0�   AsP   As!|P   WC�=As!��   As!|P   As#�P   V޲�As$�   As#�P   As&^P   W���As&��   As&^P   As(�P   Wʛ�As(��   As(�P   As+@P   W���As+e�   As+@P   As-�P   W��`As-��   As-�P   As0"P   Wi��As0G�   As0"P   As2�P   W��.As2��   As2�P   As5P   W셕As5)�   As5P   As7uP   X~�