CDF   �   
      time       bnds      lon       lat          2   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �CMCC-CM2-SR5 (2016): 
aerosol: MAM3
atmos: CAM5.3 (1deg; 288 x 192 longitude/latitude; 30 levels; top at ~2 hPa)
atmosChem: none
land: CLM4.5 (BGC mode)
landIce: none
ocean: NEMO3.6 (ORCA1 tripolar primarly 1 deg lat/lon with meridional refinement down to 1/3 degree in the tropics; 362 x 292 longitude/latitude; 50 vertical levels; top grid cell 0-1 m)
ocnBgchem: none
seaIce: CICE4.0      institution       QFondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici, Lecce 73100, Italy      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    comment       none   contact       	T. Lovato      creation_date         2020-05-27T09:06:37Z   data_specs_version        01.00.31   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Ohttps://furtherinfo.es-doc.org/CMIP6.CMCC.CMCC-CM2-SR5.historical.none.r1i1p1f1    grid      native atmosphere regular grid     
grid_label        gn     history      8Wed Nov 09 19:00:40 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.historical.r1i1p1f1.Amon.rlut.gn.v20200616/rlut_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20200616.fldsum.nc /Users/benjamin/github/d22a-mcdc/data/sum_mul_area_yearmean_mergetime/rlut/CMCC-CM2-SR5_r1i1p1f1/rlut_CMCC-CM2-SR5_r1i1p1f1_historical.mergetime.nc
Fri Nov 04 02:57:31 2022: cdo -O -s -fldsum -setattribute,rlut@units=W m-2 m2 -mul -yearmean -selname,rlut /Users/benjamin/Data/p22b/CMIP6/rlut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.historical.r1i1p1f1.Amon.rlut.gn.v20200616/rlut_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22b/CMIP6/areacella/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.historical.r1i1p1f1.fx.areacella.gn.v20200616/areacella_fx_CMCC-CM2-SR5_historical_r1i1p1f1_gn.nc /Users/benjamin/Data/p22c/CMIP6/sum_mul_area_yearmean/rlut/CMCC-CM2-SR5_r1i1p1f1/CMIP6.CMIP.CMCC.CMCC-CM2-SR5.historical.r1i1p1f1.Amon.rlut.gn.v20200616/rlut_Amon_CMCC-CM2-SR5_historical_r1i1p1f1_gn_185001-201412.yearmean.mul.areacella_historical_v20200616.fldsum.nc
2020-05-27T09:06:37Z ;rewrote data to be consistent with CMIP for variable rlut found in table Amon.;
none   initialization_index            institution_id        CMCC   mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      CMCC-CM2-SR5   parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      
references        none   run_variant       1st realization    	source_id         CMCC-CM2-SR5   source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(05 February 2020) MD5:6a248fd76c55aa6d6f7b3cc6866b5faf      title         &CMCC-CM2-SR5 output prepared for CMIP6     variable_id       rlut   variant_label         r1i1p1f1   license      ?CMIP6 model data produced by CMCC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https:///pcmdi.llnl.gov/. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.5.0      tracking_id       1hdl:21.14100/0eea0845-d6ae-4f8e-8e54-1126d9a4a008      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      365_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                   	   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2 m2   
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         FLUT   cell_measures         area: areacella             �                Aq���   Aq��P   Aq�P   [ڱ�Aq�6�   Aq�P   Aq��P   [��Aq���   Aq��P   Aq��P   [�/Aq��   Aq��P   Aq�dP   [���Aq���   Aq�dP   Aq��P   [��]Aq���   Aq��P   Aq�FP   [��YAq�k�   Aq�FP   Aq��P   [�Aq���   Aq��P   Aq�(P   [���Aq�M�   Aq�(P   Aq��P   [�	Aq���   Aq��P   Aq�
P   [��'Aq�/�   Aq�
P   Aq�{P   [ڷFAq���   Aq�{P   Aq��P   [ڋ5Aq��   Aq��P   Aq�]P   [�̨AqĂ�   Aq�]P   Aq��P   [�z$Aq���   Aq��P   Aq�?P   [�$Aq�d�   Aq�?P   Aq˰P   [�9�Aq���   Aq˰P   Aq�!P   [��Aq�F�   Aq�!P   AqВP   [�c�Aqз�   AqВP   Aq�P   [�f�Aq�(�   Aq�P   Aq�tP   [�DAqՙ�   Aq�tP   Aq��P   [�9hAq�
�   Aq��P   Aq�VP   [ڣ�Aq�{�   Aq�VP   Aq��P   [���Aq���   Aq��P   Aq�8P   [ڠmAq�]�   Aq�8P   Aq�P   [�ۚAq���   Aq�P   Aq�P   [ک�Aq�?�   Aq�P   Aq�P   [�xAq��   Aq�P   Aq��P   [�Y�Aq�!�   Aq��P   Aq�mP   [ڏ�Aq��   Aq�mP   Aq��P   [�|�Aq��   Aq��P   Aq�OP   [�`�Aq�t�   Aq�OP   Aq��P   [ژ�Aq���   Aq��P   Aq�1P   [��Aq�V�   Aq�1P   Aq��P   [١�Aq���   Aq��P   Aq�P   [��Aq�8�   Aq�P   Aq��P   [��vAq���   Aq��P   Aq��P   [�6uAq��   Aq��P   ArfP   [���Ar��   ArfP   Ar�P   [���Ar��   Ar�P   ArHP   [���Arm�   ArHP   Ar�P   [�ƜAr��   Ar�P   Ar*P   [��kArO�   Ar*P   Ar�P   [��Ar��   Ar�P   ArP   [ڑ�Ar1�   ArP   Ar}P   [��8Ar��   Ar}P   Ar�P   [�P`Ar�   Ar�P   Ar_P   [ٶ�Ar��   Ar_P   Ar�P   [���Ar��   Ar�P   ArAP   [�#bArf�   ArAP   Ar�P   [�9Ar��   Ar�P   Ar!#P   [�iAr!H�   Ar!#P   Ar#�P   [��Ar#��   Ar#�P   Ar&P   [�e�Ar&*�   Ar&P   Ar(vP   [�r�Ar(��   Ar(vP   Ar*�P   [��6Ar+�   Ar*�P   Ar-XP   [�ݼAr-}�   Ar-XP   Ar/�P   [��pAr/��   Ar/�P   Ar2:P   [�5�Ar2_�   Ar2:P   Ar4�P   [��Ar4��   Ar4�P   Ar7P   [ڣ�Ar7A�   Ar7P   Ar9�P   [���Ar9��   Ar9�P   Ar;�P   [�[�Ar<#�   Ar;�P   Ar>oP   [٩Ar>��   Ar>oP   Ar@�P   [�ukArA�   Ar@�P   ArCQP   [��WArCv�   ArCQP   ArE�P   [��ArE��   ArE�P   ArH3P   [ڝyArHX�   ArH3P   ArJ�P   [ڞ�ArJ��   ArJ�P   ArMP   [�a7ArM:�   ArMP   ArO�P   [ډ�ArO��   ArO�P   ArQ�P   [�E'ArR�   ArQ�P   ArThP   [��ArT��   ArThP   ArV�P   [���ArV��   ArV�P   ArYJP   [�R�ArYo�   ArYJP   Ar[�P   [�:Ar[��   Ar[�P   Ar^,P   [�-�Ar^Q�   Ar^,P   Ar`�P   [�7�Ar`��   Ar`�P   ArcP   [ڬ9Arc3�   ArcP   AreP   [ڇ�Are��   AreP   Arg�P   [ڌ�Arh�   Arg�P   ArjaP   [�y�Arj��   ArjaP   Arl�P   [�#*Arl��   Arl�P   AroCP   [�9�Aroh�   AroCP   Arq�P   [�_�Arq��   Arq�P   Art%P   [�D�ArtJ�   Art%P   Arv�P   [ڡGArv��   Arv�P   AryP   [ڧ�Ary,�   AryP   Ar{xP   [�|�Ar{��   Ar{xP   Ar}�P   [��Ar~�   Ar}�P   Ar�ZP   [� >Ar��   Ar�ZP   Ar��P   [���Ar���   Ar��P   Ar�<P   [�O9Ar�a�   Ar�<P   Ar��P   [�I�Ar���   Ar��P   Ar�P   [��rAr�C�   Ar�P   Ar��P   [��Ar���   Ar��P   Ar� P   [ڇYAr�%�   Ar� P   Ar�qP   [�9�Ar���   Ar�qP   Ar��P   [�CAr��   Ar��P   Ar�SP   [���Ar�x�   Ar�SP   Ar��P   [ڥ2Ar���   Ar��P   Ar�5P   [�G Ar�Z�   Ar�5P   Ar��P   [��fAr���   Ar��P   Ar�P   [��iAr�<�   Ar�P   Ar��P   [ڟAr���   Ar��P   Ar��P   [�"�Ar��   Ar��P   Ar�jP   [�8�Ar���   Ar�jP   Ar��P   [��rAr� �   Ar��P   Ar�LP   [�n]Ar�q�   Ar�LP   Ar��P   [���Ar���   Ar��P   Ar�.P   [���Ar�S�   Ar�.P   Ar��P   [��`Ar���   Ar��P   Ar�P   [�PAr�5�   Ar�P   Ar��P   [�BAr���   Ar��P   Ar��P   [��OAr��   Ar��P   Ar�cP   [�5Ar���   Ar�cP   Ar��P   [�{kAr���   Ar��P   Ar�EP   [�?.Ar�j�   Ar�EP   ArĶP   [ك�Ar���   ArĶP   Ar�'P   [��Ar�L�   Ar�'P   ArɘP   [�Arɽ�   ArɘP   Ar�	P   [�8{Ar�.�   Ar�	P   Ar�zP   [ڐArΟ�   Ar�zP   Ar��P   [�=Ar��   Ar��P   Ar�\P   [�˨ArӁ�   Ar�\P   Ar��P   [ِ?Ar���   Ar��P   Ar�>P   [فAr�c�   Ar�>P   ArگP   [���Ar���   ArگP   Ar� P   [�)Ar�E�   Ar� P   ArߑP   [� �Ar߶�   ArߑP   Ar�P   [�QAr�'�   Ar�P   Ar�sP   [چ�Ar��   Ar�sP   Ar��P   [ڔ�Ar�	�   Ar��P   Ar�UP   [�6�Ar�z�   Ar�UP   Ar��P   [�"FAr���   Ar��P   Ar�7P   [�+�Ar�\�   Ar�7P   Ar�P   [��Ar���   Ar�P   Ar�P   [ٳ�Ar�>�   Ar�P   Ar��P   [�QAr���   Ar��P   Ar��P   [�e$Ar� �   Ar��P   Ar�lP   [�H Ar���   Ar�lP   Ar��P   [�?Ar��   Ar��P   Ar�NP   [ڐAr�s�   Ar�NP   As�P   [�@As��   As�P   As0P   [��AsU�   As0P   As�P   [�/YAs��   As�P   As	P   [�l�As	7�   As	P   As�P   [َ�As��   As�P   As�P   [�ܴAs�   As�P   AseP   [���As��   AseP   As�P   [��OAs��   As�P   AsGP   [��GAsl�   AsGP   As�P   [�`(As��   As�P   As)P   [��WAsN�   As)P   As�P   [�ʄAs��   As�P   AsP   [�bSAs0�   AsP   As!|P   [�+�As!��   As!|P   As#�P   [�b~As$�   As#�P   As&^P   [��As&��   As&^P   As(�P   [ڷ�As(��   As(�P   As+@P   [���As+e�   As+@P   As-�P   [�P�As-��   As-�P   As0"P   [ڋ�As0G�   As0"P   As2�P   [��As2��   As2�P   As5P   [ڌ*As5)�   As5P   As7uP   [�T�