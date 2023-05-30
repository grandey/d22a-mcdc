CDF   �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.2.1 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-02-20T02:44:35Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.historical.none.r1i1p1f1   grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      MTue May 30 16:58:27 2023: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsdt.gn.v20190222/rsdt_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190603.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rsdt/MRI-ESM2-0_r1i1p1f1/rsdt_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 05:44:43 2022: cdo -O -s -fldsum -setattribute,rsdt@units=W m-2 m2 -mul -yearmean -selname,rsdt /Users/benjamin/Data/p22b/CMIP6/rsdt/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsdt.gn.v20190222/rsdt_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.fx.areacella.gn.v20190603/areacella_fx_MRI-ESM2-0_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rsdt/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rsdt.gn.v20190222/rsdt_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20190603.fldsum.nc
2019-02-20T02:44:35Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_historical_101 (sfc_avr_mon.ctl)      initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rsdt   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/9a6d670f-496a-45e0-bb49-500662c4fb49      CDO       @Climate Data Operators version 2.2.0 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rsdt                   
   standard_name         toa_incoming_shortwave_flux    	long_name          TOA Incident Shortwave Radiation   units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       9Shortwave radiation incident at the top of the atmosphere      original_name         DSWT   cell_measures         area: areacella    history       r2019-02-20T02:44:35Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).              �e+20).              �                Aq���   Aq��P   Aq�P   \) Aq�6�   Aq�P   Aq��P   \'�Aq���   Aq��P   Aq��P   \(�Aq��   Aq��P   Aq�dP   \%ZAq���   Aq�dP   Aq��P   \"/Aq���   Aq��P   Aq�FP   \ =Aq�k�   Aq�FP   Aq��P   \ �Aq���   Aq��P   Aq�(P   \!Aq�M�   Aq�(P   Aq��P   \$�Aq���   Aq��P   Aq�
P   \(Aq�/�   Aq�
P   Aq�{P   \+�Aq���   Aq�{P   Aq��P   \*Aq��   Aq��P   Aq�]P   \'MAqĂ�   Aq�]P   Aq��P   \%�Aq���   Aq��P   Aq�?P   \%Aq�d�   Aq�?P   Aq˰P   \"�Aq���   Aq˰P   Aq�!P   \"Aq�F�   Aq�!P   AqВP   \!Aqз�   AqВP   Aq�P   \$bAq�(�   Aq�P   Aq�tP   \'ZAqՙ�   Aq�tP   Aq��P   \'�Aq�
�   Aq��P   Aq�VP   \*Aq�{�   Aq�VP   Aq��P   \)�Aq���   Aq��P   Aq�8P   \&�Aq�]�   Aq�8P   Aq�P   \$�Aq���   Aq�P   Aq�P   \!5Aq�?�   Aq�P   Aq�P   \ �Aq��   Aq�P   Aq��P   \�Aq�!�   Aq��P   Aq�mP   \=Aq��   Aq�mP   Aq��P   \�Aq��   Aq��P   Aq�OP   \!�Aq�t�   Aq�OP   Aq��P   \%Aq���   Aq��P   Aq�1P   \$)Aq�V�   Aq�1P   Aq��P   \#�Aq���   Aq��P   Aq�P   \)�Aq�8�   Aq�P   Aq��P   \%wAq���   Aq��P   Aq��P   \!�Aq��   Aq��P   ArfP   \qAr��   ArfP   Ar�P   \�Ar��   Ar�P   ArHP   \:Arm�   ArHP   Ar�P   \Ar��   Ar�P   Ar*P   \$�ArO�   Ar*P   Ar�P   \*�Ar��   Ar�P   ArP   \-:Ar1�   ArP   Ar}P   \1�Ar��   Ar}P   Ar�P   \-�Ar�   Ar�P   Ar_P   \(�Ar��   Ar_P   Ar�P   \#�Ar��   Ar�P   ArAP   \":Arf�   ArAP   Ar�P   \ 'Ar��   Ar�P   Ar!#P   \�Ar!H�   Ar!#P   Ar#�P   \�Ar#��   Ar#�P   Ar&P   \YAr&*�   Ar&P   Ar(vP   \!�Ar(��   Ar(vP   Ar*�P   \*Ar+�   Ar*�P   Ar-XP   \%�Ar-}�   Ar-XP   Ar/�P   \+9Ar/��   Ar/�P   Ar2:P   \(Ar2_�   Ar2:P   Ar4�P   \+)Ar4��   Ar4�P   Ar7P   \&IAr7A�   Ar7P   Ar9�P   \#yAr9��   Ar9�P   Ar;�P   \_Ar<#�   Ar;�P   Ar>oP   \�Ar>��   Ar>oP   Ar@�P   \)ArA�   Ar@�P   ArCQP   \!EArCv�   ArCQP   ArE�P   \*IArE��   ArE�P   ArH3P   \2�ArHX�   ArH3P   ArJ�P   \6HArJ��   ArJ�P   ArMP   \7IArM:�   ArMP   ArO�P   \/�ArO��   ArO�P   ArQ�P   \)�ArR�   ArQ�P   ArThP   \$rArT��   ArThP   ArV�P   \ oArV��   ArV�P   ArYJP   \�ArYo�   ArYJP   Ar[�P   \"DAr[��   Ar[�P   Ar^,P   \&Ar^Q�   Ar^,P   Ar`�P   \,�Ar`��   Ar`�P   ArcP   \1�Arc3�   ArcP   AreP   \/cAre��   AreP   Arg�P   \,Arh�   Arg�P   ArjaP   \,�Arj��   ArjaP   Arl�P   \'�Arl��   Arl�P   AroCP   \$�Aroh�   AroCP   Arq�P   \"8Arq��   Arq�P   Art%P   \$ArtJ�   Art%P   Arv�P   \*Arv��   Arv�P   AryP   \8Ary,�   AryP   Ar{xP   \8Ar{��   Ar{xP   Ar}�P   \5|Ar~�   Ar}�P   Ar�ZP   \4�Ar��   Ar�ZP   Ar��P   \1{Ar���   Ar��P   Ar�<P   \,�Ar�a�   Ar�<P   Ar��P   \(�Ar���   Ar��P   Ar�P   \$dAr�C�   Ar�P   Ar��P   \&�Ar���   Ar��P   Ar� P   \-�Ar�%�   Ar� P   Ar�qP   \1aAr���   Ar�qP   Ar��P   \;�Ar��   Ar��P   Ar�SP   \@iAr�x�   Ar�SP   Ar��P   \;�Ar���   Ar��P   Ar�5P   \4NAr�Z�   Ar�5P   Ar��P   \*ZAr���   Ar��P   Ar�P   \*�Ar�<�   Ar�P   Ar��P   \&�Ar���   Ar��P   Ar��P   \&�Ar��   Ar��P   Ar�jP   \-Ar���   Ar�jP   Ar��P   \;�Ar� �   Ar��P   Ar�LP   \G�Ar�q�   Ar�LP   Ar��P   \IAr���   Ar��P   Ar�.P   \A�Ar�S�   Ar�.P   Ar��P   \=�Ar���   Ar��P   Ar�P   \2&Ar�5�   Ar�P   Ar��P   \*wAr���   Ar��P   Ar��P   \'�Ar��   Ar��P   Ar�cP   \'�Ar���   Ar�cP   Ar��P   \'UAr���   Ar��P   Ar�EP   \-Ar�j�   Ar�EP   ArĶP   \4cAr���   ArĶP   Ar�'P   \9.Ar�L�   Ar�'P   ArɘP   \8�Arɽ�   ArɘP   Ar�	P   \90Ar�.�   Ar�	P   Ar�zP   \2�ArΟ�   Ar�zP   Ar��P   \3PAr��   Ar��P   Ar�\P   \+�ArӁ�   Ar�\P   Ar��P   \(�Ar���   Ar��P   Ar�>P   \(�Ar�c�   Ar�>P   ArگP   \(�Ar���   ArگP   Ar� P   \.~Ar�E�   Ar� P   ArߑP   \8�Ar߶�   ArߑP   Ar�P   \A�Ar�'�   Ar�P   Ar�sP   \@�Ar��   Ar�sP   Ar��P   \@�Ar�	�   Ar��P   Ar�UP   \6�Ar�z�   Ar�UP   Ar��P   \6ZAr���   Ar��P   Ar�7P   \+�Ar�\�   Ar�7P   Ar�P   \&�Ar���   Ar�P   Ar�P   \']Ar�>�   Ar�P   Ar��P   \)�Ar���   Ar��P   Ar��P   \5Ar� �   Ar��P   Ar�lP   \B	Ar���   Ar�lP   Ar��P   \?�Ar��   Ar��P   Ar�NP   \?Ar�s�   Ar�NP   As�P   \;jAs��   As�P   As0P   \/<AsU�   As0P   As�P   \(�As��   As�P   As	P   \']As	7�   As	P   As�P   \%�As��   As�P   As�P   \(GAs�   As�P   AseP   \2�As��   AseP   As�P   \:�As��   As�P   AsGP   \AnAsl�   AsGP   As�P   \>"As��   As�P   As)P   \?�AsN�   As)P   As�P   \1zAs��   As�P   AsP   \,GAs0�   AsP   As!|P   \%EAs!��   As!|P   As#�P   \#�As$�   As#�P   As&^P   \!IAs&��   As&^P   As(�P   \!�As(��   As(�P   As+@P   \ �As+e�   As+@P   As-�P   \%�As-��   As-�P   As0"P   \-�As0G�   As0"P   As2�P   \1�As2��   As2�P   As5P   \2�As5)�   As5P   As7uP   \4�